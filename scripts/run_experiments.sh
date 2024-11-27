#!/bin/bash


if test ! -d build_dir ; then
  mkdir build_dir && cd build_dir &&
  cmake -DDEAL_II_DIR=$DEAL_II_DIR .. && make -j4 
else
# Compile code
  cd build_dir &&
  cmake -DDEAL_II_DIR=$DEAL_II_DIR .. && make -j4 
fi


declare -a array=("../parameters/circle.prm" 
                  "../parameters/square.prm" 
                  "../parameters/flower.prm")


echo -e "\n******RUNNING THE EXPERIMENTS******\n******RESULTS WILL BE SAVED UNDER build_dir/ DIRECTORY******\n" &&
for prm_file in "${array[@]}"
do
 ./block_demo $prm_file | tee codimension_1_different_geometries.out 
 echo -e "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
done
echo -e "******DONE. RESULTS SAVED AT /build_dir/codimension_1_different_geometries.out******"