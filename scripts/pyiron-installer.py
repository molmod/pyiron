#! /usr/bin/python
# coding: utf-8

import os,sys
from zipfile import ZipFile
from shutil import copytree, rmtree
import stat
import urllib.request as urllib2
from pathlib import Path

def write_environ_var(full_path,var,path,replace=False,prepend="",append=""):
    """
        Write the environment variable (var) equal to full_path+path
        If it already exists either append it (default) or replace it (replace=True)
    """
    new_path = os.path.join(full_path,path)
    if var in os.environ:
        # check if the environment variable already contains the path we are giving
        environ_paths = os.environ[var].split(':')
        if new_path in environ_paths:
            pass
        else:
            new_bashrc = ""
            old_var = "export {}={}".format(var,os.environ[var])
            if replace:
                new_var = "export {}={}".format(var,"{}".format(new_path))
            else:
                new_var = "export {}={}".format(var,os.environ[var]+":{}".format(new_path))
            new_var = prepend+new_var+append

            with open(os.path.expanduser("~/.bashrc"), "r") as bashrc:
                for line in bashrc:
                    line = line.rstrip()
                    changes = line.replace(old_var, new_var)
                    new_bashrc += changes + "\n"

            st = os.stat(os.path.expanduser("~/.bashrc"))
            with open(os.path.expanduser("~/.bashrc"), "w") as bashrc:
                bashrc.write(new_bashrc)
            os.chmod(os.path.expanduser("~/.bashrc"), st.st_mode)

    else:
        # add paths
        with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
            bashrc.write(prepend + "export {}={}\n".format(var,new_path) + append)


def write_full_environ_var(env_loc=None,location='~/'):
    if env_loc is None:
        full_path = os.path.normpath(os.path.abspath(os.path.expanduser(location)))
        download_path = full_path
    else:
        full_path = '$' + env_loc + '/' + location # this should be correctly expanded by the shell
        download_path = os.environ[env_loc]+'/'+location

    write_environ_var(full_path,'PYIRONPROJECTPATHS','pyiron/projects',prepend='\n')
    # replace the resource_paths, since it will automatically read the first
    write_environ_var(full_path,'PYIRONRESOURCEPATHS','pyiron/resources',replace=True,append='\n')

    print('Please source the .bashrc or reset your terminal.')
    print('')
    print('$ source ~/.bashrc')

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

    with ZipFile(temp_zip_file) as zip_file_object:
        zip_file_object.extractall(temp_directory)
    copytree(temp_extract_folder, user_directory, dirs_exist_ok=True)
    if os.name != "nt":  #
        for root, dirs, files in os.walk(user_directory):
            for file in files:
                if ".sh" in file:
                    st = os.stat(os.path.join(root, file))
                    os.chmod(os.path.join(root, file), st.st_mode | stat.S_IEXEC)
    os.remove(temp_zip_file)
    rmtree(temp_extract_folder)

################################
# MAIN

location = sys.argv[1]
try:
    folder = sys.argv[2]
except IndexError:
    folder = None

# Write the environment variables or append them with new paths
download_path = write_full_environ_var(env_loc=None,location=location)

# Download resources and overwrite them if they already exist
download_resources(zip_file="resources.zip",
                   resource_directory=os.path.join(download_path,'pyiron/resources'),
                   projects_directory=os.path.join(download_path,'pyiron/projects'),
                   giturl_for_zip_file="https://github.com/SanderBorgmans/pyiron-resources/archive/hpc_ugent.zip",
                   git_folder_name="pyiron-resources-hpc_ugent")

# Create the specified folder
if folder is not None:
    Path(os.path.join(download_path,'pyiron/projects',folder)).mkdir(parents=True, exist_ok=True)
