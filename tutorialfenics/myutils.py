def log(message, logfile):
    # print(message)
    logfile.write(message + "\n")
    logfile.flush()