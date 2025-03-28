# Code for adding new lines into output file of this app.
# It's hard to read without =)
import textwrap

input_file = 'Transcription/Lectures/20-02.txt'
output_file = '20-02_with_new_lines.txt'
max_line_length = 120

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        line = line.strip()
        if not line:
            f.write('\n')  # preserve empty lines
            continue
        wrapped = textwrap.wrap(line, width=max_line_length)
        for subline in wrapped:
            f.write(subline + '\n')

        f.write('\n')
