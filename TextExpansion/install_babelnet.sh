#!/bin/bash

BABELNET_PATH="http://lasid.sor.ufscar.br/textexpansion/complete_babelnet.tar.gz"
BASEDIR=$(dirname $0)

function download_babelnet {
    #mkdir ~/BabelNet
    cd ~

    echo "Downloading BabelNet"
    wget -c $BABELNET_PATH
}

echo "This script installs BabelNet in your system. The following file needs to be downloaded:"
echo "$BABELNET_PATH"
echo "This script can download this file or you can download it manually and place it under your Home directory."
echo "Do you want to download this file now? [y/n]"
read -r resp

if [ $resp == 'y' ]
then
    download_babelnet
else
    echo "From this point, this script assumes you have downloaded the file mentioned before and placed it under your Home directory."
    echo "Do you want to continue? [y/n]"
    read -r inner_resp

    if [ $inner_resp != 'y' ]
    then
        exit 1
    fi
fi

cd ~

echo "Extracting BabelNet"

tar -xzvf complete_babelnet.tar.gz

cd BabelNet/babelnet-api-1.0.1/config

echo "Configuring BabelNet"

rm babelnet.var.properties
echo "babelnet.dir=$HOME/BabelNet/babelnet_paths" > babelnet.var.properties

rm knowledge.var.properties
echo "knowledge.graph.pathIndex=$HOME/BabelNet" > knowledge.var.properties

echo "All done!"
read -r resp

