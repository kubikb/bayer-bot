from __future__ import print_function

__author__ = 'kubikbalint@gmail.com (Balint Kubik)'

import sys

from oauth2client import client
from googleapiclient import sample_tools
import nltk
from random import randint
from textgenrnn import textgenrnn
import re

# Modify this to get a new post with the defined image
PREFIX_TO_USE = "Jobbik"
IMG_URL = "http://www.atv.hu/thumbnail/772x514/f/25/d2000.jpg"

# Download NLTK punkt to enable sentence tokenization
nltk.download('punkt')

# Model generates whitespaces before punctuation marks. Fix this behavior.
def fix_whitespace_before_punctuation(text):
	return re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)

# Generate one paragraph of text
def get_paragraph(model, prefix, max_num_chars=1000, temperature=0.3):
	raw_text = model.generate(
		temperature=temperature,
		max_gen_length=max_num_chars,
		prefix=prefix,
		return_as_list=True
	)[0]

	# Remove last sentence since it will be mostly incomplete
	sentences = nltk.sent_tokenize(raw_text)[:-1]

	final_text = fix_whitespace_before_punctuation(" ".join(sentences))
	print("Generated paragraph containing %s characters" % (len(final_text)))

	return final_text


def main(argv):
	# Authenticate and construct service.
	service, flags = sample_tools.init(
		argv, 'blogger', 'v3', __doc__, __file__,
		scope='https://www.googleapis.com/auth/blogger')

	try:
		# Load model
		model = textgenrnn(
			weights_path="bayer_bot_weights.hdf5",
			vocab_path="bayer_bot_vocab.json",
			config_path="bayer_bot_config.json"
		)
		
		# Randomly generate paragraphs with varying lengths
		num_paragraphs = randint(2,10)
		print("Generating %s paragraphs..." % num_paragraphs)

		paragraphs = []

		for parag_id in range(1, num_paragraphs + 1):
			num_max_chars = randint(500, 2000)
			print("Paragraph #%s will have %s characters." %(parag_id, num_max_chars))

			paragraph_text = get_paragraph(
				model, PREFIX_TO_USE, max_num_chars=num_max_chars, temperature=0.3
			)

			paragraphs.append(paragraph_text)

		# Generate HTML content for the blog post
		img_part = "<div class=\"separator\" style=\"clear: both; text-align: center;\"><a href=\"http://www.atv.hu/thumbnail/772x514/f/25/d2000.jpg\" imageanchor=\"1\" style=\"margin-left: 1em; margin-right: 1em;\"><img border=\"0\" src=\"%s\" max-height: 100px;/></a></div><br />" % IMG_URL

		paragraph_divs = "\n".join(
			["<div>	%s</div>" % text for text in paragraphs]
		)

		blog_body = "<div style =\"text-align: justify;\">%s %s</div>" %(img_part, paragraph_divs)

		# Find out our blog id
		blog_to_post_to = service.blogs().getByUrl(url="https://bayerbot.blogspot.com/").execute()
		blog_id = blog_to_post_to["id"]
		
		# Post to blog
		posts = service.posts()
		req = posts.insert(
			blogId=blog_id,
			isDraft=False,
			body={
				"kind": "blogger#post",
				"title": PREFIX_TO_USE.upper(),
				"content": blog_body
			}
		).execute()

		# Print the result of the API call
		print(req)

	except client.AccessTokenRefreshError:
		print ('The credentials have been revoked or expired, please re-run'
		   'the application to re-authorize')

if __name__ == '__main__':
	main(sys.argv)
