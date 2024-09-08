#!/usr/bin/env bash
# pass-through commands to 'docker run' with some defaults
# https://docs.docker.com/engine/reference/commandline/run/
ROOT="$(dirname "$(readlink -f "$0")")"

# check for V4L2 devices
V4L2_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	# give docker root user X11 permissions
	sudo xhost +si:localuser:root
	
	# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
	XAUTH=/tmp/.docker.xauth
	xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
	chmod 777 $XAUTH

	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# extra flags
EXTRA_FLAGS=""

if [ -n "$HUGGINGFACE_TOKEN" ]; then
	EXTRA_FLAGS="$EXTRA_FLAGS --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN"
fi

# check if sudo is needed
if [ $(id -u) -eq 0 ] || id -nG "$USER" | grep -qw "docker"; then
	SUDO=""
else
	SUDO="sudo"
fi

# run the container
ARCH=$(uname -m) #i

if [ $ARCH = "aarch64" ]; then

	# this file shows what Raspberry PI 5 board is running
	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/rpi_model  

	set -x
	$SUDO docker run -d --rm --network host \
                --privileged \
				--volume /tmp:/tmp \
				--volume /var/log:/var/log \
				--volume /etc:/etc \
				--volume /home/pi:/home/pi \
				--volume $ROOT/config.yml:/home/config.yml \
				--volume /var/run/dbus:/var/run/dbus \
				--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
				--volume /var/run/docker.sock:/var/run/docker.sock \
				--device /dev/snd \
				--device /dev/bus/usb \
				$DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $EXTRA_FLAGS \
				"$@"

elif [ $ARCH = "x86_64" ]; then

	set -x

	$SUDO docker run -d --rm --network host \
				--privileged \
				--volume /tmp:/tmp \
				--volume /var/log:/var/log \
				--volume /etc:/etc \
				--volume /home/pi:/home/pi \
				--volume $ROOT/config.yml:/home/config.yml \
				--volume /var/run/dbus:/var/run/dbus \
				--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
				--volume /var/run/docker.sock:/var/run/docker.sock \
				--device /dev/snd \
				--device /dev/bus/usb \
				$DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $EXTRA_FLAGS \
				"$@"
fi
