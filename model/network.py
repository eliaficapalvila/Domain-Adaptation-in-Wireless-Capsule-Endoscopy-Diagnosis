import tensorflow as tf

from SenseTheFlow.async import Model, DataParser, EvalCallbackHook
from SenseTheFlow.layers import WNAdam, utils
from SenseTheFlow import config

from resnet_TL.resnet_model import Model as ResnetModel
from resnet_TL.resnet_model import get_block_sizes, conv2d_fixed_padding, batch_norm

from loss import pairwise_distances, batch_all_triplet_loss, batch_hard_triplet_loss


class Network(object):
    
    def __init__(self, params, reader):
        
        self.params = params
        self.reader = reader
        
        self.num_samples = params['num-samples']
        
        self.results = {key: [] for key in self.params['fetch-tensors']}
               
    """
    Parser function: to read and preprocessing the images
    """
    def parser_train(self, feature, label, mode):        
        return feature, label

    def parser_eval(self, feature, label, mode):
        return feature, label
    
    
    """
    To return confusion matrix
    """
    def eval_confusion_matrix(self,labels, predictions):
        with tf.variable_scope("eval_confusion_matrix"):
            con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=self.params['num-classes'])

            con_matrix_sum = tf.Variable(tf.zeros(shape=(self.params['num-classes'],self.params['num-classes']), 
                                                  dtype=tf.int32),
                                                trainable=False,
                                                name="confusion_matrix_result",
                                                collections=[tf.GraphKeys.LOCAL_VARIABLES])


            update_op = tf.assign_add(con_matrix_sum, con_matrix)

            return tf.convert_to_tensor(con_matrix_sum), update_op
        
    """
    Function to retrieve tf.identities
    """
    def step(self, model, results, k, step):
        for key, value in results.items():
            self.results[key].append(value)
        
        
    """
    Network Graph
    """
    def cnn_model_fn(self, features, labels, mode, params = {}):
                
        assert 'num-classes' in  params, ' No num_classes'
        assert 'embedding-size' in params, ' No embedding-size'
        assert 'triplet-strategy' in params, ' No triplet-strategy'
        
        width_size, height_size, n_channels = params['image-shape']
        
        if not mode == tf.estimator.ModeKeys.PREDICT:
            labels = tf.reshape(labels, shape = [-1])
            labels = tf.cast(labels, tf.int32)
    
        params['is-training'] = (mode == tf.estimator.ModeKeys.TRAIN)
        if not params.get('data-format'):
            params['data-format'] = utils.detect_data_format()
        
        resnet_size = params.get('resnet-size', 50)
        resnet_model = ResnetModel(
            resnet_size=resnet_size,
            bottleneck=True,
            num_classes=params['num-classes'],
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            final_size=params['embedding-size'],
            resnet_version=2,
            data_format=params['data-format'],
            skip_dense=True,
            skip_reduction = False,
            dtype=tf.float32
        )
        
        """
        Computation of the graph
        """
        with tf.device(params['gpu']):
            images = utils.to_data_format(features, 'channels_last', params['data-format'])
            
            if params.get('with-images'):
                tf.summary.image('train_image', features, max_outputs=5)
            
            resnet_outputs = resnet_model(images, params['is-training'])
            
            with tf.variable_scope('feature_extractor'):
                feature_map = resnet_outputs['reduce']
                embeddings = tf.reshape(feature_map, [-1, params['embedding-size']])
                
                embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
                tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
    
                if params.get('train-only-last-layer'):
                    feature_map = tf.stop_gradient(feature_map)
                    
            with tf.variable_scope('classifier'):
                feature_map = tf.reshape(feature_map, [-1, params['embedding-size']])
                logits = tf.layers.dense(inputs=feature_map, units= params['num-classes']) 
            
    
            """
                Losses defition
            """
            with tf.variable_scope('losses'): 
            
                triplet_loss = 0
                if params.get('include-triplet-loss'):
                    with tf.variable_scope('triplet_loss'):
                        
                        margin_strategy = False if params['triplet-margin-strategy'] == 'hinge-margin' else True
                                                
                        if params['triplet-strategy'] == 'batch-all':
                            triplet_loss, fraction, _ = batch_all_triplet_loss(pairwise_distances, labels, embeddings, softplus = margin_strategy, margin = params['margin'])

                        elif params['triplet-strategy'] == 'batch-hard':
                            triplet_loss, _ = batch_hard_triplet_loss(pairwise_distances, labels, embeddings, softplus = margin_strategy, margin = params['margin'])

                            
                        if params.get('with-scalars'):
                            tf.summary.scalar('triplet_loss', triplet_loss)
                            
                l2_loss = 0
                l2_loss_scalar = 0.1
                if params.get('include-l2-loss'):
                    with tf.variable_scope('l2_loss'):
                        # Function to avoid batch_normalization in l2 loss
                        def exclude_batch_norm(name):
                            return 'batch_normalization' not in name

                        loss_filter_fn = exclude_batch_norm

                        l2_loss = params['weight-decay'] * tf.add_n(
                            # loss is computed using fp32 for numerical stability.
                            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if loss_filter_fn(v.name)])

                        l2_loss = l2_loss_scalar * l2_loss
                        if params.get('with-scalars'):
                            tf.summary.scalar('l2_loss', l2_loss)
                            
                classification_loss = 0
                with tf.variable_scope('classification'):
                    probabilities = tf.nn.softmax(logits, name = 'softmax_tensor_anchor')
                    class_pred = tf.argmax(logits, axis=1)
                    accuracy = tf.metrics.accuracy(labels, class_pred)
                                       
                    classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                    
                    tf.summary.scalar('accuracy', accuracy[1])
                    tf.summary.scalar('loss/classification', classification_loss)
                    
                            
                loss = triplet_loss + l2_loss + classification_loss

                
            """
            Metrics
            """
            with tf.variable_scope("metrics"):
                if params['include-triplet-loss']:
                    if params['triplet-strategy'] == "batch-all":
                        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm), 
                                           'metrics/loss/clf' : tf.metrics.mean(classification_loss),
                                           'metrics/loss/triplet-loss' : tf.metrics.mean(triplet_loss),
                                               'metrics/accuracy/clf': accuracy}
                    elif params['triplet-strategy'] == "batch-hard":
                        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm), 
                                           'metrics/loss/clf' : tf.metrics.mean(classification_loss),
                                           'metrics/loss/triplet-loss' : tf.metrics.mean(triplet_loss),
                                               'metrics/accuracy/clf': accuracy}
                    else:
                        raise ValueError("Triplet strategy not recognized: {}".format(params['triplet-strategy']))

                    if params['triplet-strategy'] == "batch-all":
                        eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)
                else:
                    eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm), 
                                       'metrics/loss/clf' : tf.metrics.mean(classification_loss),
                                           'metrics/accuracy/clf': accuracy}

            tf.identity(self.eval_confusion_matrix(labels, class_pred),'confusion_matrix')
            
            tf.identity(labels, 'labels')
            tf.identity(embeddings, 'embeddings')
            tf.identity(probabilities, 'probabilities')
            tf.identity(class_pred, 'predictions')
            
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = eval_metric_ops)
            

            """ 
            Optimizer 
            """
            with tf.variable_scope('optimizer'):
                global_step = tf.train.get_or_create_global_step()
                if params['optimizer'] == 'Adam':
                    optimizer = tf.train.AdamOptimizer(params['learning-rate'])
                elif params['optimizer'] == 'SGD':
                    optimizer = tf.train.GradientDescentOptimizer(params['learning-rate'])
                elif params['optimizer'] == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(params['learning-rate'])
                elif params['optimizer'] == 'MomentumOptimizer':
                    optimizer = tf.train.MomentumOptimizer(params['learning-rate'])
                else:
                    raise ValueError("Change optimizer to Adam, SGD, MomentumOptimizer or Adagrad.")
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss = loss, global_step=global_step)

            return tf.estimator.EstimatorSpec(
                mode, 
                loss=loss, 
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )
                       
    
    def run_model(self, wait = False, only_train = False, save_results = False):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        model_dir = self.params['model-dir']
        print(model_dir)

        ws = None
        warm_dir = self.params['warm-dir']
        if warm_dir:
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=warm_dir, vars_to_warm_start="resnet_model\/.*")
        
    
        with Model(model_fn = self.cnn_model_fn, 
                   model_dir = model_dir, 
                   config = config, 
                   warm_start_from = ws,  
                   params = self.params, 
                   delete_existing = self.params['delete-existing'],
                   prepend_timestamp = False, 
                   append_timestamp = False) as model:

            tf.logging.set_verbosity(tf.logging.ERROR)

            data_parser = DataParser()


            data_parser.train_from_generator(
                parser_fn = self.parser_train,
                generator = lambda: self.reader.iterate_train(),
                output_types = (tf.float32, tf.int32), #check
                output_shapes = ([None, None, 3], [1]), #crec que be
                pre_shuffle = None,
                post_shuffle = None,
                flatten = False,
                num_samples = None,
                batch_size = self.params['batch-size'],
                prefetch = self.params['prefetch'],
            )

            data_parser.eval_from_generator(
                parser_fn = self.parser_eval,
                generator = lambda: self.reader.iterate_eval(),
                output_types=(tf.float32, tf.int32),
                output_shapes = ([None, None, 3], [1]),
                pre_shuffle=False,
                post_shuffle=False,
                flatten=False,
                num_samples = self.params.get('num-samples') ,
                batch_size=1
            )
            
            

            model.data(data_parser)

            eval_callback = EvalCallbackHook(
                aggregate_callback=None,
                step_callback = self.step,
                fetch_tensors = self.params['fetch-tensors']
            )

            eval_callback = eval_callback if save_results else None
            
            self._train = model.train(self.params['train-epochs'], self.params['eval-epochs'])
            
            if wait:
                self._train.wait()

    def eval_model(self, save_results = False):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        model_dir = self.params['model-dir']
        print(model_dir)
            
        with Model(self.cnn_model_fn, model_dir, 
                   config=config, 
                   params=self.params, 
                   delete_existing = False, 
                   prepend_timestamp=False, 
                   append_timestamp=False) as model:

            tf.logging.set_verbosity(tf.logging.ERROR)

            data_parser = DataParser()

            data_parser.eval_from_generator(
                parser_fn = self.parser_eval,
                generator = lambda: self.reader.iterate_eval(),
                output_types=(tf.float32, tf.int32),
                output_shapes = ([None, None, 3], [1]),
                pre_shuffle=False,
                post_shuffle=False,
                flatten=False,
                num_samples = None, 
                batch_size=1
            )

            model.data(data_parser)

            eval_callback = EvalCallbackHook(
                aggregate_callback=None,
                step_callback = self.step,
                fetch_tensors = self.params['fetch-tensors']
            )

            eval_callback = eval_callback if save_results else None
            
            exc = model.evaluate(1, eval_callback=eval_callback)
            exc.wait()
            return exc     
