#!/bin/bash

inpFile=$1
outFile=$2
langId=$3
randNum=$4

basePath=$5
# basePath=/var/www/html/IITM_TTS/E2E_TTS_FS2/text_proc/text_norm

mapFile=charmap/charmap_${langId}.txt
tempFile=.text_temp_${randNum}.txt

cp $inpFile $tempFile

sed -i 's/ \+/ /g' $tempFile
sed -i 's/^ \+//g' $tempFile
sed -i 's/ \+$//g' $tempFile

sed -i 's/#//g' $tempFile
sed -i 's/[\.,;।] /# /g' $tempFile
sed -i 's/[?;:)(!|&’‘,।\."]//g' $tempFile
sed -i "s:[/']::g" $tempFile
sed -i 's/[-–]/ /g' $tempFile

if [[ -f "${basePath}/${mapFile}" ]]; then
	cat ${basePath}/${mapFile} | parallel -j1 -k --colsep '\t' "sed -i 's/{1}/ {2} /g' ${tempFile}"
fi

sed -i 's/ \+/ /g' $tempFile
sed -i 's/^ \+//g' $tempFile
sed -i 's/ \+$//g' $tempFile
sed -i 's/#$//g' $tempFile
sed -i 's/# \+$//g' $tempFile

awk -v var=$randNum '{printf "Python_%s_%04d %s\n",var,NR,$0}' $tempFile > $outFile
