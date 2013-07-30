
'''
Configure:
python setup.py build

StructuredModels
Colin Lea
2013
'''

from distutils.core import setup
# from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# ext_modules = [
# 				Extension("pyKinectTools_algs_Dijkstras", ["pyKinectTools/algs/dijkstras.pyx"],language='c++'),
# 				Extension("pyKinectTools_algs_local_occupancy_pattern", ["pyKinectTools/algs/LocalOccupancyPattern.pyx"],language='c++'),
# 				]
#
# for e in ext_modules:
# 	e.pyrex_directives = {
# 						"boundscheck": False,
# 						"wraparound": False,
# 						"infer_types": True
# 						}
# 	e.extra_compile_args = ["-w"]

setup(
	author = 'Colin Lea',
	author_email = 'colincsl@gmail.com',
	description = '',
	license = "FreeBSD",
	version= "0.1",
	name = 'StructuredModels',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	packages= [	"StructuredModels",
				"StructuredModels.models",
				],
	# package_data={'':['*.xml', '*.png', '*.yml', '*.txt']},
	# ext_modules = ext_modules
)

