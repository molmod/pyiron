#!/bin/bash

echo "Welcome to the automatic pyiron installer."
echo "For now this installer provides an automatic procedure for either PhD students or students following the courses:"

courses=("phd" "nano" "atmol")

echo "    * Modelling and Engineering of Nanoscale Materials (nano)"
echo "    * Atomic and Molecular Physics (atmol)"


# Allow a selection of the course or phd setting
select name in ${courses[@]}; do
    if [ -z "$name" ]
    then
        echo "You did not select a valid option. You can exit this installation prompt through CTRL+D."
    else
        echo "The installation process will now start for: $name"
        break
    fi
done

# Make a symbolic link for non phd students
# In the case of nano and atmol courses, remove any existing resource folder for a clean resource install
case $name in
    phd)
        location=$VSC_SCRATCH_KYUKON_VO_USER
        folder=""
        ;;

    nano)
        location=$VSC_SCRATCH_KYUKON
        folder="nanoscale"
        if [ ! -L "scratch" ]; then
            ln -s $location scratch
        fi
        if [ -d "scratch/pyiron/resources" ]; then
            echo "Removing existing resource folder and replacing it with a fresh install." 
            rm -r scratch/pyiron/resources
        fi
        ;;

    atmol)
        location=$VSC_SCRATCH_KYUKON
        folder="atmol"
        if [ ! -L "scratch" ]; then
            ln -s $location scratch
        fi
        if [ -d "scratch/pyiron/resources" ]; then
            echo "Removing existing resource folder and replacing it with a fresh install." 
            rm -r scratch/pyiron/resources
        fi
        ;;
esac

pyironlocation="$location"

# Install pyiron at pyironlocation
pyiron-installer.py $pyironlocation $folder

# Activate VASP potentials for phd students
case $name in
    phd)
        cd "$pyironlocation/pyiron/resources/vasp/potentials"
        bash link_potentials.sh
        cd ~
        ;;
esac

# Configure jupyter notebook
jupyter-nbextension install widgetsnbextension --py --user
jupyter-nbextension enable widgetsnbextension --py --user
jupyter-nbextension install nglview --py --user
jupyter-nbextension enable nglview --py --user
pip install jupyter_nbextensions_configurator
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user

echo "Installation complete!"
