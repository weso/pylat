class DataSample():
    def __init__(self, msg_id, msg_text, board_id,
                 root_id, kudos, author_id, post_time,
                 edit_time, thread_id, parent_msg_id,
                 views, label):
        self.msg_id = msg_id
        self.msg_text = msg_text
        self.board_id = board_id
        self.root_id = root_id
        self.kudos = kudos
        self.author_id = author_id
        self.post_time = post_time
        self.edit_time = edit_time
        self.thread_id = thread_id
        self.parent_msg_id = parent_msg_id
        self.views = views
        self.label = label

    def to_df_data(self):
        return {
            'id': self.msg_id,
            'text': self.msg_text,
            'board': self.board_id,
            'root': self.root_id,
            'kudos': self.kudos,
            'author_id': self.author_id,
            'post_time': self.post_time,
            'edit_time': self.edit_time,
            'thread': self.thread_id,
            'parent': self.parent_msg_id,
            'views': self.views,
            'label': self.label
        }
