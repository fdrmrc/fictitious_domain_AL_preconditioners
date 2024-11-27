#!/bin/bash


if test ! -e block_demo.cc ; then
  echo "You must run this script from the top level directory of the project!"
  exit
  mkdir build_dir && cd build_dir &&
  cmake .. && make -j4 
else
# Compile code
  cd build_dir &&
  cmake .. && make -j4 
fi


declare -a array=("../parameters/circle.prm" 
                  "../parameters/square.prm" 
                  "../parameters/flower.prm")


echo -e "\n******RUNNING THE EXPERIMENTS******\n"
echo -e "******OUTPUT WILL BE SAVED UNDER build_dir/ DIRECTORY******\n"
for prm_file in "${array[@]}"
do
 ./block_demo $prm_file | tee codimension_1_different_geometries.out 
 echo -e "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
done
echo -e "******DONE. RESULTS SAVED AT /build_dir/codimension_1_different_geometries.out******"
echo -e "******ITERATION COUNTS SAVED AT /build_dir/dofs_and_iterations.csv******"
