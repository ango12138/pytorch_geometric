import torch_geometric.datasets.utils  # noqa

from .actor import Actor
from .airfrans import AirfRANS
from .airports import Airports
from .amazon import Amazon
from .amazon_products import AmazonProducts
from .aminer import AMiner
from .aqsol import AQSOL
from .attributed_graph_dataset import AttributedGraphDataset
from .ba2motif_dataset import BA2MotifDataset
from .ba_shapes import BAShapes
from .bitcoin_otc import BitcoinOTC
from .citation_full import CitationFull, CoraFull
from .coauthor import Coauthor
from .coma import CoMA
from .dblp import DBLP
from .dbp15k import DBP15K
from .deezer_europe import DeezerEurope
from .dgraph import DGraphFin
from .dynamic_faust import DynamicFAUST
from .elliptic import EllipticBitcoinDataset
from .email_eu_core import EmailEUCore
from .entities import Entities
from .explainer_dataset import ExplainerDataset
from .facebook import FacebookPagePage
from .fake import FakeDataset, FakeHeteroDataset
from .faust import FAUST
from .flickr import Flickr
from .freebase import FB15k_237
from .gdelt import GDELT
from .ged_dataset import GEDDataset
from .gemsec import GemsecDeezer
from .geometry import GeometricShapes
from .github import GitHub
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .hgb_dataset import HGBDataset
from .hydro_net import HydroNet
from .icews import ICEWS18
from .imdb import IMDB
from .infection_dataset import InfectionDataset
from .jodie import JODIEDataset
from .karate import KarateClub
from .last_fm import LastFM
from .lastfm_asia import LastFMAsia
from .linkx_dataset import LINKXDataset
from .lrgb import LRGBDataset
from .malnet_tiny import MalNetTiny
from .md17 import MD17
from .mixhop_synthetic_dataset import MixHopSyntheticDataset
from .mnist_superpixels import MNISTSuperpixels
from .modelnet import ModelNet
from .molecule_net import MoleculeNet
from .movie_lens import MovieLens
from .nell import NELL
from .ogb_mag import OGB_MAG
from .omdb import OMDB
from .pascal import PascalVOCKeypoints
from .pascal_pf import PascalPF
from .pcpnet_dataset import PCPNetDataset
from .planetoid import Planetoid
from .polblogs import PolBlogs
from .ppi import PPI
from .qm7 import QM7b
from .qm9 import QM9
from .reddit import Reddit
from .reddit2 import Reddit2
from .rel_link_pred_dataset import RelLinkPredDataset
from .s3dis import S3DIS
from .sbm_dataset import (
    RandomPartitionGraphDataset,
    StochasticBlockModelDataset,
)
from .shapenet import ShapeNet
from .shrec2016 import SHREC2016
from .snap_dataset import SNAPDataset
from .suite_sparse import SuiteSparseMatrixCollection
from .taobao import Taobao
from .tosca import TOSCA
from .tu_dataset import TUDataset
from .twitch import Twitch
from .upfd import UPFD
from .webkb import WebKB
from .wikics import WikiCS
from .wikipedia_network import WikipediaNetwork
from .willow_object_class import WILLOWObjectClass
from .word_net import WordNet18, WordNet18RR
from .yelp import Yelp
from .zinc import ZINC

__all__ = [
    'KarateClub',
    'TUDataset',
    'GNNBenchmarkDataset',
    'Planetoid',
    'FakeDataset',
    'FakeHeteroDataset',
    'NELL',
    'CitationFull',
    'CoraFull',
    'Coauthor',
    'Amazon',
    'PPI',
    'Reddit',
    'Reddit2',
    'Flickr',
    'Yelp',
    'AmazonProducts',
    'QM7b',
    'QM9',
    'MD17',
    'ZINC',
    'AQSOL',
    'MoleculeNet',
    'Entities',
    'RelLinkPredDataset',
    'GEDDataset',
    'AttributedGraphDataset',
    'MNISTSuperpixels',
    'FAUST',
    'DynamicFAUST',
    'ShapeNet',
    'ModelNet',
    'CoMA',
    'SHREC2016',
    'TOSCA',
    'PCPNetDataset',
    'S3DIS',
    'GeometricShapes',
    'BitcoinOTC',
    'ICEWS18',
    'GDELT',
    'DBP15K',
    'WILLOWObjectClass',
    'PascalVOCKeypoints',
    'PascalPF',
    'SNAPDataset',
    'SuiteSparseMatrixCollection',
    'AMiner',
    'WordNet18',
    'WordNet18RR',
    'FB15k_237',
    'WikiCS',
    'WebKB',
    'WikipediaNetwork',
    'Actor',
    'OGB_MAG',
    'DBLP',
    'MovieLens',
    'IMDB',
    'LastFM',
    'HGBDataset',
    'JODIEDataset',
    'MixHopSyntheticDataset',
    'UPFD',
    'GitHub',
    'FacebookPagePage',
    'LastFMAsia',
    'DeezerEurope',
    'GemsecDeezer',
    'Twitch',
    'Airports',
    'BAShapes',
    'LRGBDataset',
    'MalNetTiny',
    'OMDB',
    'PolBlogs',
    'EmailEUCore',
    'StochasticBlockModelDataset',
    'RandomPartitionGraphDataset',
    'LINKXDataset',
    'EllipticBitcoinDataset',
    'DGraphFin',
    'HydroNet',
    'ExplainerDataset',
    'InfectionDataset',
    'BA2MotifDataset',
    'AirfRANS',
    'Taobao',
]

classes = __all__
