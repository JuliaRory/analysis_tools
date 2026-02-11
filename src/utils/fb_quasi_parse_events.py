from numpy import zeros, asarray

def trigger_to_event_v1_1(trigger, window_size=600):
    """
    Parse a photodiode trigger signal to detect motor and rest events.

    This function scans a binary photomark signal and identifies events
    based on the magnitude of fluctuations within a sliding window. 
    It returns an array of the same length as the input trigger, where
    each element indicates the type of event at that time.

    Parameters
    ----------
    trigger : array-like
        Binary (0 or 1) signal from a photodiode, representing stimulus fluctuations.
    window_size : int, optional, default=600
        Number of samples to consider in the sliding window when detecting changes.

    Returns
    -------
    events : array-like
        Array of the same length as `trigger`, containing:
        - 0 : no event
        - 1 : motor event
        - 2 : rest event
    trigger_sum : array-like
        Array of the same length as `trigger`, 
        containing sum of its elements in window_size. 
    """

    events = zeros(len(trigger)) 
    
    n_motor = 0
    wait_start = True
    idx_trial_start = None
    idx_rest = None
    trigger_sum = []
    pr_v = 0
    for start_idx in range(len(trigger)):
        how_much_left = len(trigger) - window_size
        end_idx = start_idx + window_size if window_size < how_much_left else start_idx + how_much_left
        tsum = sum(trigger[start_idx:end_idx])
        trigger_sum.append(tsum)

        tsummax = max(trigger_sum[-window_size:])                   # max value for the last window_size ms
        if (pr_v == tsummax) & (tsum < pr_v):                       # if new value is smaller than previous
            if wait_start:                                         # two  bursts  -> signal of a beginning 
                wait_start = False
                idx_trial_start = start_idx
            elif not(wait_start) and (n_motor < 4):                # three bursts -> signal of a motor trial 
                n_motor += 1
                if n_motor == 4:
                    events[idx_trial_start:start_idx] = 1
                    idx_rest = start_idx
            elif not(wait_start) and (n_motor == 4):               # four bursts -> signal of a rest trial
                events[idx_rest:start_idx] = 2
                n_motor = 0
                wait_start = True

        pr_v =  tsum
    
    return events, asarray(trigger_sum) 


def reparse_trigger_v1_1(trigger, window_size=600, config_info = {"motor_trial_dur": 1200,"rest_trial_dur": 5000}):
    """
    Parse a photodiode trigger signal to detect motor and rest events.

    This function scans a binary photomark signal and identifies events
    based on the magnitude of fluctuations within a sliding window. 
    It returns an array of the same length as the input trigger, where
    each element indicates the type of event at that time.

    Parameters
    ----------
    trigger : array-like
        Binary (0 or 1) signal from a photodiode, representing stimulus fluctuations.
    window_size : int, optional, default=600
        Number of samples to consider in the sliding window when detecting changes.
    config_info: dict, optional, defalt={"motor_trial_dur": 1200,"rest_trial_dur": 5000}
        motor_trial_dur : int, optional, default=1200
            Duration of a motor trial in milliseconds. Used to mark motor events.
        rest_trial_dur : int, optional, default=5000
            Duration of a rest trial in milliseconds. Used to mark rest events.

    Returns
    -------
    events : array-like
        Array of the same length as `trigger`, containing:
        - 0 : no event
        - 1 : motor event
        - 2 : rest event
    """
    events = zeros(len(trigger)) 
    motor_trial_dur = config_info["motor_trial_dur"]
    rest_trial_dur = config_info["rest_trial_dur"]

    n_motor = 0
    wait_start = True
    trigger_sum = []
    pr_v = 0
    for start_idx in range(len(trigger)):
        how_much_left = len(trigger) - window_size
        end_idx = start_idx + window_size if window_size < how_much_left else start_idx + how_much_left
        tsum = sum(trigger[start_idx:end_idx])
        trigger_sum.append(tsum)
        tsummax = max(trigger_sum[-window_size:])                   # max value for the last window_size ms
        if (pr_v == tsummax) & (tsum < pr_v):                       # if new value is smaller than previous
            if wait_start:   # signal of a start
                wait_start = False
            elif not(wait_start) and (n_motor < 4):                # signal of a motor trial
                events[start_idx-motor_trial_dur:start_idx] = 1
                n_motor += 1
            elif not(wait_start) and (n_motor == 4):               # signal of a rest trial
                events[start_idx-rest_trial_dur:start_idx] = 2
                n_motor = 0
                wait_start = True
        pr_v =  tsum
    
    return events