import random
import math

import pandas as pd
import numpy as np


# NOTE: For XSPNs we might need to remap the domain to a reduced interval beginning at 0!
class Domain:
    def __init__(self, column, integral):
        self.min = min(column)
        self.max = max(column)
        self.integral = integral

    def sample(self):
        if self.integral:
            return random.randint(self.min, self.max)
        else:
            return random.random() * (self.max - self.min) + self.min

    def sample_reduced(self, new_size):
        s = (self.sample() - self.min) / new_size
        return int(s) if self.integral else s


class DomainConversion:
    def __init__(self, min_value, max_value, offset, stretch_factor, is_intergral):
        self.min_value = min_value
        self.max_value = max_value
        self.offset = offset
        self.stretch_factor = stretch_factor
        self.is_integral = is_intergral

    def convert(self, value):
        if self.is_integral:
            return math.floor((value - self.offset) / self.stretch_factor)
        else:
            return math.floor((value - self.offset) / self.stretch_factor)

    def unconvert(self, value):
        if self.is_integral:
            return math.floor(value * self.stretch_factor) + self.offset
        else:
            return value * self.stretch_factor + self.offset

    def sample(self, space='original'):
        if space == 'original':
            val = random.random() * (self.max_value - self.min_value) + self.min_value
            return int(val) if self.is_integral else val
        elif space == 'stretched':
            upper = math.floor((self.max_value - self.offset) / self.stretch_factor)
            val = random.random() * upper
            return int(val) if self.is_integral else val
        else:
            raise ValueError(f'unknown space {space}')

    def __str__(self):
        if self.is_integral:
            upper = math.floor((self.max_value - self.offset) / self.stretch_factor)
            return f'[{self.min_value}, {self.max_value}] (integral) -> [0, {upper}] (integral)'
        else:
            upper = math.floor((self.max_value - self.offset) / self.stretch_factor)
            return f'[{self.min_value}, {self.max_value}] (real) -> [0, {upper}] (integral)'


def remap_domain(column, max_size, is_integral) -> DomainConversion:
    min_value = min(column)
    max_value = max(column)
    offset = min_value
    # > 1 if the domain is reduced in size
    if is_integral:
        stretch_factor = max((max_value - min_value) / max_size, 1)
    else:
        stretch_factor = (max_value - min_value) / max_size

    return DomainConversion(min_value, max_value, offset, stretch_factor, is_integral)


def generate_domain_mapping(data: pd.DataFrame, attribute_types: dict, domain_size=256) -> dict:
    domains = { key: remap_domain(data[f'line_item_sanitized.{key}'], domain_size,
                                  data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                for key in attribute_types.keys() }

    return domains


def remap_data(data: pd.DataFrame, attribute_types: dict):
    domains = { key: remap_domain(data[f'line_item_sanitized.{key}'], 256,
                                  data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                for key in attribute_types.keys() }

    df = data.copy(deep=True)

    #print(df)

    for attr_name in domains.keys():
        col_name = f'line_item_sanitized.{attr_name}'
        df[col_name] = df[col_name].map(domains[attr_name].convert)

    return df


def remap_data_without_types(data: pd.DataFrame, max_size: int) -> pd.DataFrame:
    df = data.copy(deep=True)

    for column in df:
        domain = remap_domain(df[column], max_size, False)
        df[column] = df[column].map(domain.convert)


    return df
