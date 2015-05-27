#!/bin/bash
#
# simply displays the current GPU usage and memory consumption every
# second on the console, or in genmon bar style when started with
# --genmon
#
# works only on Nvidia cards
#
 
export GPU=0
export MEM=0

function get_mem
{
    MEM=`nvidia-smi -q -g 0 2>&1 | grep -A 2 -i utilization | grep -i memory | tail -1 | awk '{print $3}' | sed s/\%//g`
}

function get_gpu
{
    GPU=`nvidia-smi -q -g 0 2>&1 | grep -A 2 -i utilization | grep -i gpu | tail -1 | awk '{print $3}' | sed s/\%//g`
}
 
function genmon_mode
{
    get_gpu
    get_mem
 
    if [ ${GPU} -lt 10 ]; then
        GPU=0${GPU}
    fi
 
    if [ ${MEM} -lt 10 ]; then
        MEM=0${MEM}
    fi
 
    echo "<img>/usr/share/pixmaps/nvidia-current-settings.png</img>"
    echo "<txt><b><big><tt>${GPU} ${MEM}</tt></big></b></txt>"
    echo "<tool>GPU: ${GPU}%, Memory: ${MEM}%</tool>"
}
 
function console_mode
{
    echo -e "\033[30;42mPress Ctrl-C to break ...\033[0m"
 
    while true; do
        get_gpu
        get_mem
 
        ## gpu usage
        echo -ne "\033[5G"
        echo -ne "\033[1;37;41mGPU:\033[0m ${GPU} %"
 
        ## gpu memory
        echo -ne "\033[15G"
        echo -e "\033[1;37;41mMEM:\033[0m ${MEM} %"
 
        sleep 1
    done
}
 
if [ $# -eq 0 ]; then
    console_mode
else
    case $1 in
        "--genmon") genmon_mode;;
        "--console") console_mode;;
    esac
fi
