# from distutils.core import setup
# import py2exe
# import sys
#
# if len(sys.argv) == 1:
#     sys.argv.append("py2exe")
#
# setup(options = {"py2exe": {"includes": ["matplotlib", "numpy", "pandas"]}},
#       windows=[{'script': "Graph_Drawer.pyw"}])

# from distutils.core import setup
# import py2exe
# import matplotlib
#
# setup(windows=['Graph_Drawer.pyw'],
#       options={
#                'py2exe': {"includes": ["matplotlib", "numpy", "pandas"],
#                           'packages' :  ['matplotlib', 'pytz'],
#                           'dll_excludes': ['libgdk-win32-2.0-0.dll',
#                                          'libgobject-2.0-0.dll',
#                                          'libgdk_pixbuf-2.0-0.dll',
#                                          'libgtk-win32-2.0-0.dll',
#                                          'libglib-2.0-0.dll',
#                                          'libcairo-2.dll',
#                                          'libpango-1.0-0.dll',
#                                          'libpangowin32-1.0-0.dll',
#                                          'libpangocairo-1.0-0.dll',
#                                          'libglade-2.0-0.dll',
#                                          'libgmodule-2.0-0.dll',
#                                          'libgthread-2.0-0.dll',
#                                          'QtGui4.dll', 'QtCore.dll',
#                                          'QtCore4.dll'
#                                         ],
#                           }
#                 },
#       data_files=matplotlib.get_py2exe_datafiles(),)

from distutils.core import setup
import py2exe

from distutils.filelist import findall
import os
import matplotlib

matplotlibdatadir = matplotlib.get_data_path()
matplotlibdata = findall(matplotlibdatadir)
matplotlibdata_files = []
for f in matplotlibdata:
    dirname = os.path.join('matplotlibdata', f[len(matplotlibdatadir) + 1:])
    matplotlibdata_files.append((os.path.split(dirname)[0], [f]))

setup(
    console=['Graph_Drawer.py'],
    options={
        'py2exe': {'includes': ["matplotlib.pyplot", "numpy", "pandas"],
                   'packages': ['matplotlib', 'pytz'],
                   }
    },
    data_files=matplotlibdata_files
)

# option = {
#     'bundle_files':2,
#     'compressed': True
# }
# setup(
#     options = {'py2exe': option},
#     windows =['calculater.pyw'],
#     zipfile = None,
#     )
# python setup.py
