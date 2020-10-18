import numpy as np

#ID = 0
ALPHA = 0.6
BETA = 0.6
IoU_min = 0.05
ENDPOINT = (256 / 256, 128 / 256)

class Person():
    ID = 0
    def __init__(self, idx, origin, x_vel=0, y_vel=0, a_vel=0):
        """
        id must be unique, never seen before
        
        idx points to the corresponding detection in prev_dets. A person object will not be in the people set unless it has
        a corresponding detection
        
        origin is a x1y1x2y2 vector representing the coords of the initial detection
        """
        #self.id = ID
        #ID += 1
        self.idx = idx
        self.origin = origin
        
        
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.a_vel = a_vel

def IoU(A, B):
    """
    A (m x d), B (n x d) --> C (m, n)
    
    accepts boxes in x1y1x2y2 format
    """
    ax_vec = A[:, 0][:, np.newaxis]
    ay_vec = A[:, 1][:, np.newaxis]
    aw_vec = (A[:, 2] - A[:, 0])[:, np.newaxis]
    ah_vec = (A[:, 3] - A[:, 1])[:, np.newaxis]
    
    bx_vec = B[:, 0][np.newaxis, :]
    by_vec = B[:, 1][np.newaxis, :]
    bw_vec = (B[:, 2] - B[:, 0])[np.newaxis, :]
    bh_vec = (B[:, 3] - B[:, 1])[np.newaxis, :]
    
    minw_array = np.minimum(aw_vec, bw_vec)
    wi_array = 0.5 * (aw_vec + bw_vec) - np.abs(ax_vec - bx_vec)
    wi_array[wi_array < 0] = 0
    wi_array = np.where(wi_array > minw_array, minw_array, wi_array)
    
    minh_array = np.minimum(ah_vec, bh_vec)
    hi_array = 0.5 * (ah_vec + bh_vec) - np.abs(ay_vec - by_vec)
    hi_array[hi_array < 0] = 0
    hi_array = np.where(hi_array > minh_array, minh_array, hi_array)
    
    intersection_array = wi_array * hi_array
    union_array = aw_vec * ah_vec + bw_vec * bh_vec - intersection_array
    IoU_array = intersection_array / union_array

    return IoU_array

def find_new(prev_dets, curr_dets, people, dt):
    """
    prev_dets (m x d): the previous object detections
    curr_dets (n x d): the current object detections
    dt: time passed between prev_dets and curr_dets
    people: dictionary of person objects
    
    Because T_skip = 1, we only need prev_dets and curr_dets to update objects
    """
    
    # generate pred_dets
    #pred_dets = np.copy(prev_dets)
    pred_dets = np.zeros(prev_dets.shape)
    for person in people.values():
        # calculate stuff
        x1 = prev_dets[person.idx, 0]
        y1 = prev_dets[person.idx, 1]
        x2 = prev_dets[person.idx, 2]
        y2 = prev_dets[person.idx, 3]
        
        h_temp = (y2 - y1)**2 + person.a_vel/(x2 - x1)
        pred_h = h_temp**0.5 if h_temp > 0 else 0

        w_temp = (x2 - x1)**2 + person.a_vel/(y2 - y1)
        pred_w = w_temp**0.5 if w_temp > 0 else 0
        
        #print(((x1 + x2 - pred_w)/2 + person.x_vel) - x1)
        
        pred_dets[person.idx, 0] = (x1 + x2 - pred_w)/2 + person.x_vel
        pred_dets[person.idx, 1] = (y1 + y2 - pred_h)/2 + person.y_vel
        pred_dets[person.idx, 2] = (x1 + x2 + pred_w)/2 + person.x_vel
        pred_dets[person.idx, 3] = (y1 + y2 + pred_h)/2 + person.y_vel
        
        if (x1 + x2 + pred_w)/2 + person.x_vel < (x1 + x2 - pred_w)/2 + person.x_vel:
            print("x2_pred must be bigger than x1_pred")
            raise
        if (y1 + y2 + pred_h)/2 + person.y_vel < (y1 + y2 - pred_h)/2 + person.y_vel:
            print("y2_pred must be bigger than y1_pred")
            raise
    

    IoU_array = IoU(pred_dets, curr_dets)
    
    new_people = {}
    for i in range(min(IoU_array.shape[0], IoU_array.shape[1])):
        m, n = IoU_array.shape
        idx = np.argmax(IoU_array)
        my_IoU = IoU_array[idx // n, idx % n]
        if my_IoU > IoU_min:
            prev = prev_dets[idx // n]
            curr = curr_dets[idx % n]

            # find the person object that matched to prev
            # update that person object to now map to curr
            # add that person object to new_people with updated velocity
            person = people[idx // n]
            person.idx = idx % n
            person.x_vel = (curr[0] + curr[2] - prev[0] - prev[2]) / 2
            person.y_vel = (curr[1] + curr[3] - prev[1] - prev[3]) / 2
            person.a_vel = (curr[2] - curr[0]) * (curr[3] - curr[1]) - (prev[2] - prev[0]) * (prev[3] - prev[1])
            new_people[idx % n] = person
            
            # update IoU_array
            if m >= 2 and n >= 2: # prevent an exception
                IoU_array = np.concatenate([IoU_array[:idx // n], IoU_array[(idx // n) + 1:]])
                IoU_array = np.concatenate([IoU_array[:, :idx % n], IoU_array[:, (idx % n) + 1:]], axis=1)
        else:
            # there are no more IoUs that will result in a sufficient match
            break
            
    for i in range(curr_dets.shape[0]):
        if i not in new_people.keys():
            new_people[i] = Person(i, curr_dets[i])
    
    d_count = 0
    for person in people.values():
        if person not in new_people.values():
            # decide to increment or decrement depending on conditions
            box = pred_dets[person.idx]
            x_end = (box[0] + box[2]) / 2
            y_end = (box[1] + box[3]) / 2
            area_end = (box[2] - box[0]) * (box[3] - box[1])
            end_dist = (x_end - ENDPOINT[0])**2 + (y_end - ENDPOINT[1])**2
            
            origin = person.origin
            x_start = (origin[0] + origin[2]) / 2
            y_start = (origin[1] + origin[3]) / 2
            area_start = (origin[2] - origin[0]) * (origin[3] - origin[1])
            start_dist = (x_start - ENDPOINT[0])**2 + (y_start - ENDPOINT[1])**2
            
            print("starting dist from endpoint: {}, ending dist from endpoint: {}".format(start_dist, end_dist))

            if end_dist < BETA * area_end and start_dist >= ALPHA * area_start:
                d_count += 1
            elif start_dist < ALPHA * area_start and end_dist >= BETA * area_end:
                d_count -= 1
                
    return d_count, new_people
            
            
            
            
        








