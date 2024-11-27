#!/bin/bash
set -e

if test ! -e block_demo.cc ; then
  echo "You must run this script from the top level directory of the project!"
  exit
else
# Compile code in Release mode
if [ ! -d "build_dir" ]; then
  mkdir build_dir && cd build_dir &&
  cmake -DCMAKE_BUILD_TYPE="Release" .. && make -j4 
else
  cd build_dir && make release && make -j4 
fi
fi


echo -e "\n****** RUNNING THE EXPERIMENTS IN RELEASE MODE ******\n"
echo -e "****** OUTPUT WILL BE SAVED UNDER build_dir/ DIRECTORY ******\n"

for dir in ../parameters/*/
  do
  dir=${dir%*/} 
 for prm_file in $dir/*.prm; do
    if [ -f "$prm_file" ]; then
        name=$(basename "$prm_file" .prm)
        echo "RUNNING PARAMETER FILE: $prm_file"
        ./block_demo $prm_file | tee codimension_1_$name.out 
        echo -e "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    fi
  done
done
echo -e "****** DONE. RESULTS SAVED AT /build_dir (check .out and .csv files) ******"
echo -e "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
