from setuptools import find_packages, setup

package_name = 'fpv_control_llm'

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
    maintainer='sh',
    maintainer_email='qjwdds@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_control_ddpg = fpv_control_llm.vision_control_ddpg:main',
            'vision_control_ddpg_gazebo_stable = fpv_control_llm.vision_control_ddpg_gazebo_stable:main',
            'vision_control_e2e_cnn = fpv_control_llm.vision_control_e2e_cnn:main',
            'vision_control_e2e_ResNet = fpv_control_llm.vision_control_e2e_ResNet:main',
        ],
    },
)
