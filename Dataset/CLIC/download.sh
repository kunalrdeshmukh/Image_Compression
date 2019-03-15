nohup wget https://data.vision.ee.ethz.ch/cvl/clic/professional_train.zip > /dev/null 2>&1 &
nohup wget https://data.vision.ee.ethz.ch/cvl/clic/professional_valid.zip > /dev/null 2>&1 & 
nohup wget https://data.vision.ee.ethz.ch/cvl/clic/mobile_valid.zip > /dev/null 2>&1 &
wget https://data.vision.ee.ethz.ch/cvl/clic/mobile_train.zip
sleep 10
unzip professional_train.zip
unzip professional_valid.zip
unzip mobile_valid.zip
unzip mobile_train.zip
sleep 5
rm professional_train.zip
rm professional_valid.zip
rm mobile_train.zip
rm mobile_valid.zip

