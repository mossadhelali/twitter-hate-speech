import emoji
import codecs
import sys
import csv
input_file = sys.argv[1]
output_file = sys.argv[2]
with codecs.open(input_file, 'r', encoding='utf8') as read_file:
	with codecs.open(output_file, 'w', encoding='utf8') as write_file:
		with codecs.open('meta.txt', 'w', encoding='utf8') as meta_file:
			# tsvreader = csv.reader(read_file, delimiter="\t")
	  #   	for line in tsvreader:
			row_counter = 0
			tweets = read_file.read().splitlines()
			for tweet in tweets:
				# tweet = line[]
				words = tweet.split(' ')
				# print(words)
				row_counter+= 1		
				for word,i in zip(words,range(len(words))):
					# if word has more than 1 chars and either starts or ends with emojis
					if len(word) > 1 and (word[0] in emoji.UNICODE_EMOJI or word[len(word)-1] in emoji.UNICODE_EMOJI):
						meta_file.write('on line '+str(row_counter)+'\t')
						meta_file.write('emoji word is '+ word + ' with length '+str(len(word))+'\n')
					# if len(word) > 1 and word[0] in emoji.UNICODE_EMOJI and word[len(word) - 1] in emoji.UNICODE_EMOJI:
					# 	meta_file.write('multi emoji ' +  word +  str(row_counter)+ '\n')
						new_word = ''
						for char in word:
							if char in emoji.UNICODE_EMOJI:
								new_word +=  ' ' + char + ' '
							else:
								new_word += char
						if new_word[len(new_word)-1]  == ' ':
							new_word = new_word[0:len(new_word)-1] 
						words[i] = new_word
				# print(words)
				fixed_tweet = ' '.join(words)
				# print(fixed_tweet)
				write_file.write(fixed_tweet+ '\n')
