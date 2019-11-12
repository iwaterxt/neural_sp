#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel computing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import socket
import os


def mkdir_join(path, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new direcory if the direcory does not exist.
    Args:
        path (str): path to a diretcory
        dir_name (str): a direcory name
    Returns:
        path to the new directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in range(len(dir_name)):
        # dir
        if i < len(dir_name) - 1:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        elif '.' not in dir_name[i]:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        # file
        else:
            path = os.path.join(path, dir_name[i])
    return path

def token_merge(text_in):
    text_out=""
    for char in text_in.split():
        if '<unk>' in char:
            text_out += str(char).replace('<unk><unk>', '')
        else:
            text_out += str(char) + ' '
    return text_out

def host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip