# sim/energy_model.py

def estimate_download_energy(chunk_size_kb, download_time_sec):
    """
    Estimate energy (in joules) spent downloading a chunk of size chunk_size_kb
    over download_time_sec seconds.
    """
    return 0.003 * chunk_size_kb + 0.1 * download_time_sec + 0.5

def estimate_decode_energy(bitrate_kbps):
    """
    Estimate decoding energy per second based on the bitrate.
    """
    return 0.0012 * bitrate_kbps + 0.2

