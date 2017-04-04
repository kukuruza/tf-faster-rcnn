import requests
import os, os.path as op
import urllib3
urllib3.disable_warnings()

def send_simple_message(subject, msg):

  # read api key
  apikey_path = op.join(os.getenv('CITY_PATH'), 'etc/mailgun.txt')
  with open(apikey_path) as fid:
    apikey = fid.readline().rstrip('\n')
  
  # send a message
  return requests.post(
        "https://api.mailgun.net/v3/sandbox4f3da61c7a534634b01774e0c203aafe.mailgun.org/messages",
        auth=("api", apikey),
        data={"from": "Mailgun Sandbox <postmaster@sandbox4f3da61c7a534634b01774e0c203aafe.mailgun.org>",
              "to": "Evgeny <toropov.evgeny@gmail.com>",
              "subject": subject,
              "text": msg})

        
if __name__ == '__main__':
  send_simple_message('Hi Evgeny', 'Hey this is a long message')

