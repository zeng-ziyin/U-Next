def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features'][..., :6]
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        jin = []
        feature_list = []
        sup_list = [[], [], [], []]

        for j in range(self.config.num_layers):
            if j == 0:
                feature_list = []
                for i in range(self.config.num_layers - j):
                    # encoding[(i, j-1)]
                    f_encoder_i_j = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(j) + str(i), is_training)
                    f_sampled_i_j = self.random_sample(f_encoder_i_j, inputs['sub_idx'][i])
                    feature_list.append(f_encoder_i_j)
                    feature = f_sampled_i_j
                jin.append(feature_list)

            if j > 0:
                feature_list = []
                for i in range(self.config.num_layers - j):
                    if i == 0:
                        # encoding[(i-1, j); (i-1, j+1)]
                        f_1 = jin[j-1][i]
                        f_2 = self.nearest_interpolation(jin[j-1][i+1], inputs['interp_idx'][i])
                        feature = tf.concat([f_1, f_2], axis=3)
                        if j == self.config.num_layers - 1:
                            f_3 = jin[0][i]
                            feature = tf.concat([feature, f_3], axis=3)

                    if i >= 1:
                        # encoding[(i, j-1); (i-1, j); (i-1, j+1)]
                        f_0 = feature
                        f_1 = jin[j - 1][i]
                        f_2 = self.nearest_interpolation(jin[j - 1][i + 1], inputs['interp_idx'][i])
                        feature = tf.concat([f_0, f_1, f_2], axis=3)
                        if j + i == self.config.num_layers - 1 and i != 3:
                            # encoding[(i, j-1); (i-1, j); (i-1, j+1); (0, j)]
                            f_3 = jin[0][i]
                            feature = tf.concat([feature, f_3], axis=3)

                    f_decoder_i_j = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Decoder_layer_' + str(j) + str(i), is_training)

                    f_sup = helper_tf_util.conv2d(f_decoder_i_j, self.config.num_classes, [1, 1], 'sup_' + str(i) + str(j), [1, 1], 'VALID', False, is_training, activation_fn=None)
                    f_sup = tf.squeeze(f_sup, [2])
                    sup_list[i].append(f_sup)

                    f_sampled_i_j = self.random_sample(f_decoder_i_j, inputs['sub_idx'][i])
                    feature_list.append(f_decoder_i_j)
                    feature = f_sampled_i_j

                jin.append(feature_list)

        f_layer_fc1 = helper_tf_util.conv2d(jin[-1][-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False, is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, sup_list