import faiss  


def load_index(cpuIndex):

    N_gpu = faiss.get_num_gpus()

    # # Initialize the resources
    print("Number of GPU resources", N_gpu)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)
    
    print (index.ntotal)
    
    return index
    
    
def index_read(filepath):
    
    cpuIndex = faiss.read_index(filepath)
    return cpuIndex
    
