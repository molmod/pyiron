#! /usr/bin/python
# coding: utf-8

import os,sys
from zipfile import ZipFile
from shutil import copytree, rmtree
import stat
import urllib.request as urllib2
from pathlib import Path

location = sys.argv[1]

def write_full_environ_var(env_loc=None,location='~/'):
    if env_loc is None:
        full_path = os.path.normpath(os.path.abspath(os.path.expanduser(location)))
        download_path = full_path
    else:
        full_path = '$' + env_loc + '/' + location # this should be correctly expanded by the shell
        download_path = os.environ[env_loc]+'/'+location

    # Check whether bashrc already contains PYIRON variables, this can happen if bashrc has not been sourced yet
    with open(os.path.expanduser("~/.bashrc"), "r") as outfile:
        lines = outfile.readlines()
    if not any(['PYIRON' in line for line in lines]):
        # Write to bashrc
        with open(os.path.expanduser("~/.bashrc"), "a") as outfile:
            if not lines[-1]=='\n': outfile.write('\n')
            outfile.write("# PYIRON env variables for pyiron file locations\n")
            outfile.write("export PYIRONRESOURCEPATHS={}\n".format(full_path + 'pyiron/resources'))
            outfile.write("export PYIRONPROJECTPATHS={}\n".format(full_path + 'pyiron/projects'))
        print('Please source the .bashrc after the initial configuration or reset your terminal.')
        print('')
        print('$ source ~/.bashrc')
    else:
        raise SystemError('Your .bashrc already has pyiron environment variables but has not been sourced. Please execute the following in your bash shell if you are certain these are defined correctly: source ~/.bashrc')

    return download_path


def download_resources(
    zip_file="resources.zip",
    resource_directory="~/pyiron/resources",
    projects_directory="~/pyiron/projects",
    giturl_for_zip_file="https://github.com/pyiron/pyiron-resources/archive/master.zip",
    git_folder_name="pyiron-resources-master",
):
    """
    Download pyiron resources from Github

    Args:
        zip_file (str): name of the compressed file
        resource_directory (str): directory where to extract the resources - the users resource directory
        projects_directory (str): directory created to contain all projects - the users projects directory
        giturl_for_zip_file (str): url for the zipped resources file on github
        git_folder_name (str): name of the extracted folder

    """
    # First create projects folder
    user_directory = os.path.normpath(
        os.path.abspath(os.path.expanduser(projects_directory))
    )
    Path(user_directory).mkdir(parents=True, exist_ok=True)

    # Then create resources folder and copy all relevant files
    user_directory = os.path.normpath(
        os.path.abspath(os.path.expanduser(resource_directory))
    )
    if os.path.exists(user_directory) and not os.listdir(user_directory):
        os.rmdir(user_directory)
    temp_directory = os.path.normpath(os.path.abspath(os.path.expanduser('~/tmp')))
    Path(temp_directory).mkdir(parents=True, exist_ok=True)
    temp_zip_file = os.path.join(temp_directory, zip_file)
    temp_extract_folder = os.path.join(temp_directory, git_folder_name)
    urllib2.urlretrieve(giturl_for_zip_file, temp_zip_file)
    if os.path.exists(user_directory):
        raise ValueError(
            "The resource directory exists already, therefore it can not be created: ",
            user_directory,
        )
    with ZipFile(temp_zip_file) as zip_file_object:
        zip_file_object.extractall(temp_directory)
    copytree(temp_extract_folder, user_directory)
    if os.name != "nt":  #
        for root, dirs, files in os.walk(user_directory):
            for file in files:
                if ".sh" in file:
                    st = os.stat(os.path.join(root, file))
                    os.chmod(os.path.join(root, file), st.st_mode | stat.S_IEXEC)
    os.remove(temp_zip_file)
    rmtree(temp_extract_folder)


download_path = write_full_environ_var(env_loc=None,location=location)
download_resources(zip_file="resources.zip",
                   resource_directory=os.path.join(download_path,'pyiron/resources'),
                   projects_directory=os.path.join(download_path,'pyiron/projects'),
                   giturl_for_zip_file="https://github.com/SanderBorgmans/pyiron-resources/archive/hpc_ugent.zip",
                   git_folder_name="pyiron-resources-hpc_ugent")
