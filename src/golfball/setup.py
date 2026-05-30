from setuptools import setup
import os
from glob import glob

package_name = 'golfball'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='dcardella@ucsd.edu',
    description='Golf ball ros2 package for UCSD Robocar',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drive_publisher = golfball.drive:main',
            'ball_detection_node = golfball.ball_detection_node:main',
        ],
    },
)
