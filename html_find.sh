#!/bin/sh

input=33.list
#set -x
while read line
do
        file=$(echo $line)
		result=$(grep $file index.html)
		if [ -z "$result" ]
		then
			echo $file
		fi
done < "$input"

