#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:33:21 2020

@author: giridhar
"""
import numpy as np
import sys
from numpy.lib.stride_tricks import as_strided

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError
    if overlap<0 or length<=0:
        raise ValueError

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)

def get_epoch_position_features(pms, rate, nsamples, seconds2samples=True, zero_uv_GCP=False):

    if seconds2samples:
        ## Convert seconds -> waveform sample numbers:-
        pms = np.asarray(np.round(pms * rate), dtype=int)
  
    ## make sure length compatible with the waveform:--
    last = len(pms)-1
    while pms[last] > nsamples:
        last -= 1
    pms = pms[:last]
    if nsamples > pms[-1]:
        pms = np.concatenate([pms, np.array([nsamples])])
    ## addd first 0
    pms = np.concatenate([np.array([0]), pms])
    
    start_end = segment_axis(pms, 2, overlap=1)
    lengths = start_end[:,1] - start_end[:,0]
    
    forwards = []
    backwards = []
    norm_forwards = []
    for length in lengths:
        forward = np.arange(length)
        backward = np.flipud(forward)
        norm_forward = forward / float(length)
        forwards.append( forward )
        backwards.append( backward )
        norm_forwards.append(  norm_forward )
    forwards = np.concatenate(forwards).reshape((nsamples,1))
    backwards = np.concatenate(backwards).reshape((nsamples,1))
    norm_forwards = np.concatenate(norm_forwards).reshape((nsamples,1))

    if zero_uv_GCP:
        #forwards[] = 0.0
        sys.exit('not implemented : zero_uv_GCP')
    return (forwards, backwards, norm_forwards)

def get_synthetic_pitchmarks(fz_per_sample, srate, uv_length_sec):  
    '''
    unlike in slm-local stuff, assume F0 is already upsampled, and uv regions are 0 or negative
    '''
    uv_length_samples = uv_length_sec * srate
    ## make pitch marks:
    current = 0
    pms = [current]
    while True:
        val = int(fz_per_sample[current])
        if val <= 0:
            current += uv_length_samples
        else:
            current += srate / val
                
        if current >= len(fz_per_sample):
            break
        
        current = int(current)
        
        pms.append(current)
    return np.array(pms)

def get_voicing_mask(ixx, voicing, wavelength):

    changes = (voicing[:-1] - voicing[1:])
    ons = []
    offs = []

    if voicing[0] == 1:
        ons.append(0)

    for (i,change) in enumerate(changes):
        if change < 0:
            ons.append(i)
        elif change > 0:
            offs.append(i+1)

    if voicing[-1] == 1:
        offs.append(len(voicing))

    assert len(ons) == len(offs)

    seq = np.zeros(wavelength)
    for (on, off) in zip(ons, offs):

        on_i = min(ixx[on], wavelength)
        off_i = min(ixx[off-1], wavelength)
        seq[on_i:off_i] = 1.0

    return seq


def synthesise_excitation(fzero, wavelength, srate=16000, frameshift_sec=0.005, uv_length_sec=0.005):
    fz = fzero.reshape(-1)
    mul = np.concatenate((np.ones(100),np.ones(100)*1.7), axis=0).repeat(100)
    fz = fz*mul[:len(fz)]
    fz_sample = np.repeat(fz, int(srate * frameshift_sec))
    if fz_sample.shape[0] > wavelength:
        fz_sample = fz_sample[:wavelength]
    elif fz_sample.shape[0] < wavelength:
        diff = wavelength - fz_sample.shape[0]
        fz_sample = np.concatenate([fz_sample, np.ones(diff)*fz_sample[-1]])
    
    pm = get_synthetic_pitchmarks(fz_sample, srate, uv_length_sec)
    
    f,b,sawtooth = get_epoch_position_features(pm, srate, wavelength, seconds2samples=False, zero_uv_GCP=False)

    ### TODO: refactor and merge
    fz_at_pm = fz_sample[pm]
    voicing = np.ones(pm.shape)
    voicing[fz_at_pm <= 0.0] = 0

    ## convert to 16bit range for storage later (positives only):
    halfrange = (2**16) / 2
    sawtooth *= halfrange  ## TODO: this conversion reversed a little later! rationalise....
    
    voiced_mask = get_voicing_mask(pm, voicing, wavelength)

    sawtooth = sawtooth.flatten()
    sawtooth *= voiced_mask

    return sawtooth

