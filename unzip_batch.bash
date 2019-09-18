#!/bin/bash
cd /home-4/vchandr8@jhu.edu/scratch/spec/

echo 'Made it to the directory!'

for i in 7
do
	filename="batch"$i".tar.gz"
	echo $filename
	tar -xzf $filename -C ./
done
echo 'All done'
