import os
import numpy as np
import copy
import re

RPB_KEYS = [
    ('errBias','errBias'),
    ('errRand','errRand'),
    ('LINE_OFF','lineOffset'),
    ('SAMP_OFF','sampOffset'),
    ('LAT_OFF','latOffset'),
    ('LONG_OFF','longOffset'),
    ('HEIGHT_OFF','heightOffset'),
    ('LINE_SCALE','lineScale'),
    ('SAMP_SCALE','sampScale'),
    ('LAT_SCALE','latScale'),
    ('LONG_SCALE','longScale'),
    ('HEIGHT_SCALE','heightScale'),
    ('LINE_NUM_COEFF','lineNumCoef'),
    ('LINE_DEN_COEFF','lineDenCoef'),
    ('SAMP_NUM_COEFF','sampNumCoef'),
    ('SAMP_DEN_COEFF','sampDenCoef')]

def rpc_polynomial(p,l,h,c):
    # RPC00B Definition
    f = (c[0]     + c[5]*l*h    + c[10]*p*l*h    + c[15]*(p**3)
       + c[1]*l   + c[6]*p*h    + c[11]*(l**3)   + c[16]*p*(h**2)
       + c[2]*p   + c[7]*(l**2) + c[12]*l*(p**2) + c[17]*(l**2)*h
       + c[3]*h   + c[8]*(p**2) + c[13]*l*(h**2) + c[18]*(p**2)*h
       + c[4]*l*p + c[9]*(h**2) + c[14]*(l**2)*p + c[19]*(h**3)
    )
    return f

def grad_rpc_polynomial(p,l,h,c):
    dfdp = (c[10]*l*h    + c[15]*3*(p**2)
          + c[ 6]*h      + c[16]*(h**2)
          + c[ 2]        + c[12]*2*l*p
          + c[ 8]*2*p    + c[18]*2*p*h
          + c[ 4]*l      + c[14]*(l**2)
    )
    dfdl = (c[ 5]*h      + c[10]*p*h
          + c[ 1]        + c[11]*3*(l**2)
          + c[ 7]*l      + c[12]*(p**2)   + c[17]*2*l*h
          + c[13]*(h**2)
          + c[ 4]*p      + c[14]*2*l*p
    )
    dfdh = (c[ 5]*l      + c[10]*p*l
          + c[ 6]*p      + c[16]*p*2*h
          + c[17]*(l**2)
          + c[ 3]        + c[13]*l*2*h    + c[18]*(p**2)
          + c[ 9]*2*h    + c[19]*3*(h**2)
    )
    return dfdp, dfdl, dfdh

def grad_rpc(p, l, h, num_coef, den_coef):
    num = rpc_polynomial(p,l,h, num_coef)
    den = rpc_polynomial(p,l,h, den_coef)
    dndp, dndl, dndh = grad_rpc_polynomial(p,l,h, num_coef)
    dddp, dddl, dddh = grad_rpc_polynomial(p,l,h, den_coef)
    dvdp = (dndp*den - num*dddp) / (den**2)
    dvdl = (dndl*den - num*dddl) / (den**2)
    dvdh = (dndh*den - num*dddh) / (den**2)
    return dvdp, dvdl, dvdh

class RPC:
    @classmethod
    def from_file(cls, rpc_file, imd_file=None):
        # rpc_file = fix_rpc_file(rpc_file)
        rpc = cls()
        rpc.imd = None
        rpc.cov_bias = None
        _a, ext = os.path.splitext(rpc_file)
        ext = ext.lower()
        if ext == '.rpb':
            rpc.load_rpb(rpc_file)
        else:
            NotImplementedError('Cannot load rpc model using %s file'%(ext))
        rpc.filename = rpc_file
        if (imd_file is not None):
            rpc.load_imd(imd_file)
        return rpc

    def load_rpb(self, rpb_file):
        d = load_imd(rpb_file)
        self.load_rpb_dict(d)

    def load_rpb_dict(self, d):
        rpc_coefs = {}
        p = re.compile('\(' + '([^,]+),'*19 + '([^,]+),*\)')
        for key, rpb_key in RPB_KEYS:
            if key.endswith('COEFF'):
                poly_coefs = [float(x)
                              for x in p.match(d['IMAGE'][rpb_key]).groups()]
                poly_coefs = np.array(poly_coefs)
                rpc_coefs[key] = poly_coefs
            else:
                rpc_coefs[key] = float(d['IMAGE'][rpb_key])
        rpc_coefs['satId'] = d['satId']
        rpc_coefs['SpecId'] = d['SpecId']
        rpc_coefs['bandId'] = d['bandId']
        self.load_coefs(rpc_coefs)

    def load_imd(self, imd_file):
        self.imd = load_imd(imd_file)
        d_rpb = load_imd(imd_file.replace('IMD','RPB'))
        self.rpc_coefs['errBias'] = float(d_rpb['IMAGE']['errBias'])
        self.rpc_coefs['errRand'] = float(d_rpb['IMAGE']['errRand'])

    def save_rpb(self, rpb_file):
        with open(rpb_file, 'w') as f:
            for key in ['satId', 'bandId', 'SpecId']:
                f.write('%s = %s;\n' % (key, self.rpc_coefs[key]))
            f.write('BEGIN_GROUP = IMAGE\n')
            for key, rpb_key in RPB_KEYS:
                if key.endswith('COEFF'):
                    f.write('\t%s = (\n' % rpb_key)
                    for x in self.rpc_coefs[key][:-1]:
                        f.write('\t\t\t%.15g,\n' % x)
                    f.write('\t\t\t%.15g);\n' % self.rpc_coefs[key][-1])
                else:
                    f.write('\t%s = %.15g;\n' % (rpb_key, self.rpc_coefs[key]))
            f.write('END_GROUP = IMAGE\nEND;\n')

    def load_coefs(self, rpc_coefs):
        self.rpc_coefs = rpc_coefs
        self.line_num_coef = rpc_coefs['LINE_NUM_COEFF']
        self.line_den_coef = rpc_coefs['LINE_DEN_COEFF']
        self.samp_num_coef = rpc_coefs['SAMP_NUM_COEFF']
        self.samp_den_coef = rpc_coefs['SAMP_DEN_COEFF']
        self.lat_off = rpc_coefs['LAT_OFF']
        self.lat_scale = rpc_coefs['LAT_SCALE']
        self.lon_off = rpc_coefs['LONG_OFF']
        self.lon_scale = rpc_coefs['LONG_SCALE']
        self.height_off = rpc_coefs['HEIGHT_OFF']
        self.height_scale = rpc_coefs['HEIGHT_SCALE']
        self.line_off = rpc_coefs['LINE_OFF']
        self.line_scale = rpc_coefs['LINE_SCALE']
        self.samp_off = rpc_coefs['SAMP_OFF']
        self.samp_scale = rpc_coefs['SAMP_SCALE']

    def __getitem__(self, key):
        return self.rpc_coefs[key]

    def rpc(self, lat, lon, height=None):
        # if height is None:
        #     height = get_dem_height(lat, lon, self.dem_ds)
        p = (lat - self.lat_off) / self.lat_scale
        l = (lon - self.lon_off) / self.lon_scale
        h = (height - self.height_off) / self.height_scale
        rn, cn = self.rpc_norm(p,l,h)
        line = rn*self.line_scale + self.line_off
        samp = cn*self.samp_scale + self.samp_off
        return line, samp
    
    #psuedo name
    project = rpc

    def rpc_norm(self, p, l, h):
        rn = (rpc_polynomial(p,l,h,self.line_num_coef)
              / rpc_polynomial(p,l,h,self.line_den_coef))
        cn = (rpc_polynomial(p,l,h,self.samp_num_coef)
              / rpc_polynomial(p,l,h,self.samp_den_coef))
        return rn, cn

    def affine_approx(self, lat_center, lon_center, height0, flagReturnMatrix=False):
        """Get Taylor series approx. of rpc"""
        # samp = c_aff[0] + c_aff[1]*l + c_aff[2]*p + c_aff[3]*h
        # line = c_aff[4] + c_aff[5]*l + c_aff[6]*p + c_aff[7]*h
        line_center, samp_center = self.rpc(lat_center, lon_center, height0)
        rn_center = (line_center - self.line_off) / self.line_scale
        cn_center = (samp_center - self.samp_off) / self.samp_scale

        p0 = (lat_center - self.lat_off) / self.lat_scale
        l0 = (lon_center - self.lon_off) / self.lon_scale
        h0 = (height0 - self.height_off) / self.height_scale

        drdp, drdl, drdh = grad_rpc(p0, l0, h0, self.line_num_coef, self.line_den_coef)
        dcdp, dcdl, dcdh = grad_rpc(p0, l0, h0, self.samp_num_coef, self.samp_den_coef)
        samp_aff_coef = np.array(
            [cn_center-dcdp*p0-dcdl*l0-dcdh*h0, dcdl, dcdp, dcdh])
        line_aff_coef = np.array(
            [rn_center-drdp*p0-drdl*l0-drdh*h0, drdl, drdp, drdh])

        rpc_aff_coefs = copy.deepcopy(self.rpc_coefs)
        rpc_aff_coefs['LINE_NUM_COEFF'] = np.zeros(20,dtype=float)
        rpc_aff_coefs['LINE_DEN_COEFF'] = np.zeros(20,dtype=float)
        rpc_aff_coefs['SAMP_NUM_COEFF'] = np.zeros(20,dtype=float)
        rpc_aff_coefs['SAMP_DEN_COEFF'] = np.zeros(20,dtype=float)
        rpc_aff_coefs['LINE_DEN_COEFF'][0] = 1.
        rpc_aff_coefs['SAMP_DEN_COEFF'][0] = 1.
        rpc_aff_coefs['LINE_NUM_COEFF'][:4] = line_aff_coef
        rpc_aff_coefs['SAMP_NUM_COEFF'][:4] = samp_aff_coef
        rpc_aff = RPC()
        rpc_aff.load_coefs(rpc_aff_coefs)
        rpc_aff.imd = self.imd
        rpc_aff.cov_bias = self.cov_bias
        if flagReturnMatrix:
            return rpc_aff, np.vstack((samp_aff_coef[[1,2,3,0]],
                                       line_aff_coef[[1,2,3,0]]))
        else:
            return rpc_aff

    def translate_linesamp(self, line_shift, samp_shift):
        self.line_off += line_shift
        self.rpc_coefs['LINE_OFF'] += line_shift
        self.samp_off += samp_shift
        self.rpc_coefs['SAMP_OFF'] += samp_shift


# Load the parameters from an IMD file into a dict
def load_imd(imdFile):
    with open(imdFile) as f:
        d = load_imd_fileObj(f)
    return d

def load_imd_fileObj(f):
    d = {}
    flagGroup = False
    line = ''
    for line_ in f:
        # Remove all whitespace
        line += line_.strip().replace(' ', '').replace('\t', '')
        if (not line.startswith('BEGIN_GROUP')
                and (not line.startswith('END_GROUP'))
                and (line.find(';') < 0)):
            # Keep appending until we have a ;
            continue
        if line.startswith('END;'):
            break
        # In case the line has stuff beyond the ;
        lines = line.split(';', 1)
        if len(lines) > 1:
            line, lineNext = lines
        else:
            line = lines[0]
            lineNext = ''
        var, val = line.replace(';', '').split('=')
        if flagGroup:
            if var == 'END_GROUP':
                assert groupName == val
                flagGroup = False
            else:
                d[groupName][var] = val
        else:
            if var == 'BEGIN_GROUP':
                groupName = val
                d[groupName] = {}
                flagGroup = True
            else:
                d[var] = val
        line = lineNext
    return d