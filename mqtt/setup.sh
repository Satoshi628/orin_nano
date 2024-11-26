sudo chmod 744 mqtt_sensor_control.py
# set up network
# settings->Network->PCI Ethernet->IPv4->Manual
# 192.168.1.153 255.255.255.0

# start up command
echo -e "[Unit]\nDescription=start python\nAfter=network.target\n\n[Service]\nUser=rb53262\nExecStart=/usr/bin/python3 /home/rb53262/Desktop/orin_nano/mqtt/mqtt_sensor_control.py\nType=simple\n\n[Install]\nWantedBy=multi-user.target\n" | sudo tee /etc/systemd/system/mqtt.service

sudo systemctl enable mqtt.service

# [Unit]
# ZDescription=start python
# After=network.target

# [Service]
# User=rb53262
# ExecStart=/usr/bin/python3 /home/rb53262/Desktop/orin_nano/mqtt/mqtt_sensor_control.py
# Type=simple

# [Install]
# WantedBy=multi-user.target

