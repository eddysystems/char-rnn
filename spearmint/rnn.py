import subprocess
import glob

def rnn(seq_length = None, batch_size = None, rnn_size = None, num_layers = None, dropout = None, learning_rate = None)

	# encode parameters in the filename
	savefile = "seq_length%d-batch_size%d-rnn_size%d-num_layers%d-dropout%d-learning_rate%f" % 
		    (seq_length, batch_size, rnn_size, num_layers, dropout, learning_rate)
    
	# run with these parameters. If the process bails before creating a save file, assume large objective function
	cmd = "th train.lua -seq_length %d -batch_size %d -rnn_size %d -num_layers %d -dropout %d -learning_rate %f -savefile %s -max_epochs 1 -data_dir data/javascript" %
                                        ( seq_length, batch_size, rnn_size, num_layers, dropout, learning_rate, savefile)
	subprocess.call(cmd)

	# check whether we produced any save files
	checkpoints = glob.glob("lm_%s_epoch*.t7" % savefile)

	# get the best validation loss (from filename), or a high value if there is no output
	result = 1e20
	for file in checkpoints:
		try:
    			loss = file[file.rindex('_')+1:file.rindex('.')]
		except ValueError:
			continue
		result = min(float(loss), result)
 
	print 'Result = %f' % result
	return result

# Write a function like this called 'main'
def main(job_id, params):
    print "Job %d" % job_id
    print params
    return rnn(**params)
