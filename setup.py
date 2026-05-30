from setuptools import find_packages, setup

package_name = 'main_proj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='yjhyun0613@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detect_threat_node_01 = main_proj.detect_threat_node_01:main',
            'detect_threat_02 = main_proj.detect_threat_02:main',
            'detect_threat_gui_node_01 = main_proj.detect_threat_gui_node_01:main',
            'detect_threat_node_03 = main_proj.detect_threat_node_03:main',
            'detect_threat_node_04 = main_proj.detect_threat_node_04:main',
            'detect_threat_node_05 = main_proj.detect_threat_node_05:main',
        ],
    },
)
