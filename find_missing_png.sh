#!/bin/sh

input=11.list
input2=22.list
#set -x
while read line
do
        file=$(echo $line)
		result=$(grep $file *.html)
		if [ -z "$result" ]
		then
			echo $file
		fi
done < "$input"

while read line
do
        file=$(echo $line)
		result=$(grep $file *.html)
		if [ -z "$result" ]
		then
			echo $file
		fi
done < "$input2"
