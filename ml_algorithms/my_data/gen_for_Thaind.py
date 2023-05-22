import pandas as pd
import random
import string
import time

x = ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation)
                                           for j in range(256))
print(x)
exit()

path = "/root/src/thaind/LOCA/data/recsys_data/VT/train.csv"
data = pd.read_csv(path)

print(data.head())

data = data.dropna(subset=['uid', 'sid'])

item_ids = list(set(data["sid"].to_list()))
user_ids = list(set(data["uid"].to_list()))
max_item_id = max(item_ids)
max_user_id = max(user_ids)

print("len of item_ids: ", len(item_ids))
print("Max of item_ids: ", max_item_id)
print("Min of item_ids: ", min(item_ids))
print("##"*50)
print("len of user_ids: ", len(user_ids))
print("Max of user_ids: ", max_user_id)
print("Min of user_ids: ", min(user_ids))
# exit()
# x = random.choice(item_ids)
# print(x)
# exit()

def increase_user(num_extend, cur_user_list, cur_item_list, items_per_user):
    # increase the first dimension of the sparse matrix
    max_user_id = max(cur_user_list) + 1
    new_data = pd.DataFrame(columns=['uid', 'sid'], dtype=int)
    # for uid in cur_user_list:
    #     print("Rune for user_id: ", uid)
    #     for i in range(items_per_user):
    #         temp_dict = {'user_id': uid, 'item_id': random.choice(cur_item_list)}
    #         new_data = new_data._append(temp_dict, ignore_index = True)

    print("num_extend: ", num_extend)
    for ext_uid in range(int(max_user_id), int(max_user_id) + num_extend):
        # print(ext_uid)
        temp_dict = {'uid': int(ext_uid), 'sid': int(random.choice(cur_item_list))}
        # new_data = new_data._append([ext_uid, random.choice(cur_item_list)])
        new_data = new_data._append(temp_dict, ignore_index=True)

    new_data = data._append(new_data)
    return new_data
    # exit()


ext_dim1_data = increase_user(100, user_ids, item_ids, 15)
# ext_dim1_data.to_csv('gen_data_dim1.csv', index=False)
item_ids = list(set(ext_dim1_data["sid"].to_list()))
user_ids = list(set(ext_dim1_data["uid"].to_list()))
max_item_id = max(item_ids)
max_user_id = max(user_ids)
print("New len of user_ids: ", len(user_ids))
print("New max of user_ids: ", max_user_id)
print("New min of user_ids: ", min(user_ids))
exit()


def increase_movie(num_extend, cur_user_list, cur_item_list):
    # increase the second dimension of the sparse matrix
    max_item_id = max(cur_item_list) + 1
    ext_item_list = list(range(int(max_item_id), int(max_item_id) + num_extend))
    new_data = pd.DataFrame(columns=['user_id', 'item_id'], dtype=int)
    for iid in ext_item_list:
        user_watch_movie = random.sample(cur_user_list, random.randint(1, 10))
        # user_per_item = random.randint(1, 10)
        for uid in user_watch_movie:
            temp_dict = {'user_id': int(uid), 'item_id': int(iid)}
            new_data = new_data._append(temp_dict, ignore_index=True)

    new_data = data._append(new_data)
    return new_data
    # exit()


ext_dim1_data = increase_movie(100, user_ids, item_ids)
# ext_dim1_data.to_csv('gen_data_dim1.csv', index=False)
ext_dim1_data.to_csv('data_gen_dim2.csv', index=False)
exit()

# if name==""