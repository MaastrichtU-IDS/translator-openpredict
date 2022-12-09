import os
import shutil
import time

import findspark
from gensim.models.word2vec import Word2Vec
from pyspark import SparkConf, SparkContext
from rdf2vec.walkers import RandomWalker
from sklearn.utils.validation import check_is_fitted


def walk_sequence(walker, graph, root):
    walks = walker.extract(graph, [root])
    print (walks)
    if len(walks) > 0 :
        print (len(walks))
        #walks= [list(map(str, x)) for x in walks]
        walk_strs = []
        for walk_nr, walk in enumerate(walks):
            s = ''
            for i in range(len(walk)):
                if i % 2:
                    s += '{}'.format(walk[i])
                else:
                    s += '{}'.format(walk[i])

                if i < len(walk) - 1:
                    s += '->'

            walk_strs.append(s)
        return '\n'.join(walk_strs)
    else:
        return ''


class MySentences:
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename

    def __iter__(self):
        print ('Processing ',self.filename)
        for subfname in os.listdir(self.dirname):
            if not self.filename in subfname: continue
            fpath = os.path.join(self.dirname, subfname)
            for fname in os.listdir(fpath):
                if not 'part' in fname: continue
                if '.crc' in fname: continue
                try:
                    for line in open(os.path.join(fpath, fname)):
                        line = line.rstrip('\n')
                        words = line.split("->")
                        yield words
                except Exception:
                    print("Failed reading file:")
                    print(fname)


class RDF2VecTransformer():
    """Project random walks or subtrees in graphs into embeddings, suited
    for classification.

    Parameters
    ----------
    vector_size: int (default: 500)
        The dimension of the embeddings.

    max_path_depth: int (default: 1)
        The maximum number of hops to take in the knowledge graph. Due to the
        fact that we transform s -(p)-> o to s -> p -> o, this will be
        translated to `2 * max_path_depth` hops internally.

    wl: bool (default: True)
        Whether to use Weisfeiler-Lehman embeddings

    wl_iterations: int (default: 4)
        The number of Weisfeiler-Lehman iterations. Ignored if `wl` is False.

    walks_per_graph: int (default: infinity)
        The maximum number of walks to extract from the neighborhood of
        each instance.

    n_jobs: int (default: 1)
        gensim.models.Word2Vec parameter.

    window: int (default: 5)
        gensim.models.Word2Vec parameter.

    sg: int (default: 1)
        gensim.models.Word2Vec parameter.

    max_iter: int (default: 10)
        gensim.models.Word2Vec parameter.

    negative: int (default: 25)
        gensim.models.Word2Vec parameter.

    min_count: int (default: 1)
        gensim.models.Word2Vec parameter.

    Attributes
    ----------
    model: gensim.models.Word2Vec
        The fitted Word2Vec model. Embeddings can be accessed through
        `self.model.wv.get_vector(str(instance))`.

    """
    def __init__(self, vector_size=500, walkers=RandomWalker(2, float('inf')),
                 n_jobs=1, window=5, sg=1, max_iter=10, negative=25,
                 min_count=1):
        self.vector_size = vector_size
        self.walkers = walkers
        self.n_jobs = n_jobs
        self.window = window
        self.sg = sg
        self.max_iter = max_iter
        self.negative = negative
        self.min_count = min_count

        findspark.init()

        config = SparkConf()
        config.setMaster("local[10]")
        config.set("spark.executor.memory", "70g")
        config.set('spark.driver.memory', '90g')
        config.set("spark.memory.offHeap.enabled",True)
        config.set("spark.memory.offHeap.size","50g")
        self.sc = SparkContext(conf=config)
        print (self.sc)



    def fit(self, graph, instances):
        """Fit the embedding network based on provided instances.

        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using a converter method.

        instances: array-like
            The instances for which an embedding will be created. It important
            to note that the test instances should be passed to the fit method
            as well. Due to RDF2Vec being unsupervised, there is no
            label leakage.
        -------
        """
        self.walks_ = []
        b_triples = self.sc.broadcast(graph)
        #for walker in self.walkers:
        #    self.walks_ += list(walker.extract(graph, instances))
        #print('Extracted {} walks for {} instances!'.format(len(self.walks_), len(instances)))


        folder = './walks/'
        #folder = walk_folder
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        for walker in self.walkers:
            #self.walks_ += list(walker.extract(graph, instances))
            filename = os.path.join(folder,'randwalks_n%d_depth%d_pagerank_uniform.txt'%(walker.walks_per_graph, walker.depth))
            print (filename)
            start_time =time.time()
            rdd = self.sc.parallelize(instances).map(lambda n: walk_sequence(walker, b_triples.value, n) )
            rdd.saveAsTextFile(filename)
            elapsed_time = time.time() - start_time
            print ('Time elapsed to generate features:',time.strftime("%H:%M:%S",       time.gmtime(elapsed_time)))
        print('Extracted {} walks for {} instances!'.format(len(self.walks_),
                                                            len(instances)))

        #sentences = [list(map(str, x)) for x in self.walks_]

        pattern = 'uniform'

        #vector_output =  './vectors/'
        #trainModel(entities, id2entity, walk_folder, model_folder, vector_file, pattern, maxDepth)

        sentences = MySentences(folder, filename=pattern)
        self.model_ = Word2Vec(sentences, size=self.vector_size,
                              window=self.window, workers=self.n_jobs,
                              sg=self.sg, iter=self.max_iter,
                              negative=self.negative,
                              min_count=self.min_count, seed=42)

    def transform(self, graph, instances):
        """Construct a feature vector for the provided instances.

        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using a converter method.

        instances: array-like
            The instances for which an embedding will be created. These
            instances must have been passed to the fit method as well,
            or their embedding will not exist in the model vocabulary.

        Returns
        -------
        embeddings: array-like
            The embeddings of the provided instances.
        """
        check_is_fitted(self, ['model_'])

        feature_vectors = []
        for instance in instances:
            feature_vectors.append(self.model_.wv.get_vector(str(instance)))
        return feature_vectors

    def fit_transform(self, graph, instances):
        """First apply fit to create a Word2Vec model and then generate
        embeddings for the provided instances.

        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using a converter method.

        instances: array-like
            The instances for which an embedding will be created.

        Returns
        -------
        embeddings: array-like
            The embeddings of the provided instances.
        """
        self.fit(graph, instances)
        return self.transform(graph, instances)
