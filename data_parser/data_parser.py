import pandas as pd
import datetime
import re

categorical_columns = ['message_type', 'author_type', 'severity', 'author_id']
numerical_columns = []


class DataParser:
    def __init__(self, data_path):
        print('Loading data...')
        self.data = pd.read_csv(data_path + r'\spotify-public-dataset.csv')
        print('Data cleaning and formatting...')
        self.clean_data()
        self.inject_temporal_features()
        print('Writing data...')
        self.data.to_csv(data_path + r'\modelling_data.csv', index=False)

    def clean_data(self):
        self.data.created_at = pd.to_datetime(self.data.created_at)
        self.data = self.data.drop_columns('message_id')
        self.data = self.data.dropna(subset=['message_body'])
        count = self.data.shape[0]
        timezone = self.data.created_at[0].dt.tz
        self.data = self.data[(self.data.created_at >= datetime.datetime(2017, 10, 1, tzinfo=timezone)) &
                              (self.data.created_at <= datetime.datetime(2017, 12, 5, tzinfo=timezone))]
        print('Dropped %i rows' % (int(count) - int(self.data.shape[0])))

    def inject_temporal_features(self):
        self.data['dow'] = self.data['created_at'].dt.day_name()
        self.data['month'] = self.data['created_at'].dt.month_name()
        self.data['hour'] = self.data['created_at'].dt.hour
        self.data['doy'] = self.data['created_at'].apply(lambda x: x.strftime('%j'))
        self.data['weekday'] = ['weekday' if x < 5 else 'weekend' for x in self.data['created_at'].dt.dayofweek]
        self.data['business_hours'] = ['business_hours' if (7 < x.hour < 19 and x.dayofweek < 5)
                                       else 'no_business_hours' for x in self.data['created_at']]

    def parse_handles(self):
        handle = re.compile(r'@([^\s:]+)')
        self.data['handle'] = self.data['message_body'].apply(lambda x: handle.findall(x)[0])
        self.data.loc[self.data['handle'].str.contains('SpotifyCares'), 'handle'] = 'SpotifyCares'
