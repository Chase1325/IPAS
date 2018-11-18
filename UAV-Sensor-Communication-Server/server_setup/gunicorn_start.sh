#!/bin/bash
NAME="ipas"                                                 # Name of the application
FALCONDIR=/home/pi/workspace/Cowlagi-MQP-2018/UAV-Sensor-Communication-Server               # Project directory
SOCKFILE=/home/pi/server/gunicorn-ipas.sock                   # we will communicte using this unix socket
PIDFILE=/home/pi/server/gunicorn-ipas.pid                     # Create a pid file for gunicorn process
USER=pi                                                     # the user to run as
NUM_WORKERS=$(expr 2 \* $(nproc) + 1)                       # how many worker processes should Gunicorn spawn

echo "Starting $NAME as `whoami`"

# Activate the virtual environment
cd $FALCONDIR
sudo su pi -c "cd $FALCONDIR && git pull"
source /home/pi/virtualenv/bin/activate

# Create the run directory if it doesn't exist
RUNDIR=$(dirname $SOCKFILE)
test -d $RUNDIR || mkdir -p $RUNDIR

# Start your Django Unicorn
# Programs meant to be run under supervisor should not daemonize themselves (do not use --daemon)
exec /home/pi/virtualenv/bin/gunicorn app:api \
        --name $NAME \
        --workers $NUM_WORKERS \
        --user=$USER \
        --bind=unix:$SOCKFILE \
        --log-level=debug \
        --log-file=- \
        --pid $PIDFILE
        --timeout 90

