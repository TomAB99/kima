import os, sys
import re
import io
import subprocess
from pprint import pformat
from hashlib import md5
import contextlib
import numpy as np
from . import showresults

@contextlib.contextmanager
def chdir(dir):
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


# class to interface with a kima model and a KimaResults instance
class KimaModel:
    def __init__(self):
        self.directory = os.getcwd()
        self.kima_setup = 'kima_setup.cpp'
        self.OPTIONS_file = 'OPTIONS'
        self.filename = None
        self.skip = 0
        self.units = 'kms'

        self._levels_hash = ''
        self._loaded = False

        self.obs_after_HARPS_fibers = False
        self.GP = False
        self.MA = False
        self.hyperpriors = False
        self.trend = False
        self.known_object = False

        self.fix_Np = True
        self.max_Np = 1

        self.set_priors('default')
        self._planet_priors = ('Pprior', 'Kprior', 'eprior', 'wprior', 'phiprior')

        self.threads = 1
        self.OPTIONS = {
            'particles': 2,
            'new_level_interval': 5000,
            'save_interval': 2000,
            'thread_steps': 100,
            'max_number_levels': 0,
            'lambda': 10,
            'beta': 100,
            'max_saves': 100,
            'samples_file': '',
            'sample_info_file': '',
            'levels_file': '',
        }

    # def __repr__(self):
    #     r = f'kima model\n'
    #     r += f'  directory: {self.directory}\n'
    #     r += '\n'
    #     opt_names = ('GP', 'MA', 'hyperpriors', 'trend', 'known_object')
    #     for opt in opt_names:
    #         r += f'  {opt} {getattr(self, opt)}\n'
    #     # opt = {k:v for k,v in self.__dict__.items() if k in opt_names}
    #     # r +=  pformat(opt, compact=True)
    #     r += '\n'
    #     r += f'  fix = {self.fix_Np} \t npmax = {self.max_Np}\n'
    #     r += f'  datafile = {self.filename}\n'
    #     return r

    def __str__(self):
        return f'kima model in {self.directory}'

    @property
    def multi_instrument(self):
        if self.filename is None:
            return False
        if len(self.filename) == 1:
            return False
        else:
            return True

    @property
    def _default_priors(self):
        dp = {
            # name: (default, distribution, arg1, arg2)
            'Cprior': (True, 'Uniform', self.ymin, self.ymax),
            'Jprior': (True, 'ModifiedLogUniform', 1.0, 100.0),
            'slope_prior': 
                (True, 'Uniform', -self.topslope if self.data else None, self.topslope),
            'offsets_prior': 
                (True, 'Uniform', -self.yspan if self.data else None, self.yspan),
            #
            'log_eta1_prior': (True, 'Uniform', -5.0, 5.0),
            'eta2_prior': (True, 'LogUniform', 1.0, 100.0),
            'eta3_prior': (True, 'Uniform', 10.0, 40.0),
            'log_eta4_prior': (True, 'Uniform', -1.0, 1.0),
            #
            'Pprior': (True, 'LogUniform', 1.0, 100000.0),
            'Kprior': (True, 'ModifiedLogUniform', 1.0, 1000.0),
            'eprior': (True, 'Uniform', 0.0, 1.0),
        }
        return dp

    def set_priors(self, which='default', *args):
        if which == 'default':
            self.priors = self._default_priors
        else:
            if len(args) == 3:
                assert args[1] == 'Fixed'
            else:
                assert len(args) == 4
            self.priors.update({which: args})

    def set_prior_to_default(self, which):
        default_prior = self._default_priors[which]
        self.priors.update({which: default_prior})

    @property
    def data(self):
        if self.filename is None:
            return None

        d = dict(t=[], y=[], e=[])
        for f in self.filename:
            try:
                t, y, e = np.loadtxt(f, skiprows=self.skip, usecols=range(3)).T
            except:
                return None
            d['t'].append(t)
            d['y'].append(y)
            d['e'].append(e)

        d['t'] = np.concatenate(d['t'])
        d['y'] = np.concatenate(d['y'])
        d['e'] = np.concatenate(d['e'])
        return d

    @property
    def ymin(self):
        if self.data:
            return self.data['y'].min().round(3)

    @property
    def ymax(self):
        if self.data:
            return self.data['y'].max().round(3)

    @property
    def yspan(self):
        if self.data:
            return self.data['y'].ptp().round(3)

    @property
    def topslope(self):
        if self.data:
            val = np.abs(self.data['y'].ptp() / self.data['t'].ptp()).round(3)
            return val

    def results(self, force=False):
        # calculate the hash of the levels.txt file the first time
        # we create self.res to avoid calling showresults repeatedly
        levels_f = os.path.join(self.directory, 'levels.txt')
        h = md5(open(levels_f, 'rb').read()).hexdigest()

        output = None
        if h != self._levels_hash or force:
            self._levels_hash = h
            # with io.StringIO() as buf, contextlib.redirect_stdout(buf):  # redirect stdout
            with chdir(self.directory):
                self.res = showresults(force_return=True, show_plots=False, verbose=False)
               # output = buf.getvalue()
        # self.res.return_figs = True

        return output

    def load(self):
        """
        Try to read and load a kima_setup file with the help of RegExp.
        Note that C++ is notoriously hard to parse, so for anything other than
        fairly standard kima_setup files, don't expect this function to work!
        """
        if not os.path.exists(self.kima_setup):
            raise FileNotFoundError(f'Cannot find "{self.kima_setup}"')

        setup = open(self.kima_setup).read()

        # find general model settings
        bools = (
            'obs_after_HARPS_fibers',
            'GP',
            'MA',
            'hyperpriors',
            'trend',
            'known_object',
        )
        for b in bools:
            pat = re.compile(f'const bool {b} = (\w+)')
            match = pat.findall(setup)
            if len(match) == 1:
                setattr(self, b, True if match[0] == 'true' else False)
            else:
                msg = f'Cannot find setting {b} in {self.kima_setup}'
                raise ValueError(msg)

        # find fix Np
        pat = re.compile(r'fix\((\w+)\)')
        match = pat.findall(setup)
        if len(match) == 1:
            self.fix_Np = True if match[0] == 'true' else False
        else:
            msg = f'Cannot find option for fix in {self.kima_setup}'
            raise ValueError(msg)

        # find max Np
        pat = re.compile(r'npmax\(([-+]?[0-9]+)\)')
        match = pat.findall(setup)
        if len(match) == 1:
            if int(match[0]) < 0:
                raise ValueError('npmax must be >= 0')
            self.max_Np = int(match[0])
        else:
            msg = f'Cannot find option for npmax in {self.kima_setup}'
            raise ValueError(msg)

        # find priors (here be dragons!)
        number = '([-+]?[0-9]*.?[0-9]*[E0-9]*)'
        for prior in self._default_priors.keys():
            pat = re.compile(
                rf'{prior}\s?=\s?make_prior<(\w+)>\({number}\s?,\s?{number}\)')
            match = pat.findall(setup)
            if len(match) == 1:
                m = match[0]
                dist, arg1, arg2 = m[0], float(m[1]), float(m[2])
                self.set_priors(prior, False, dist, arg1, arg2)

        # find datafile(s)
        pat = re.compile(f'const bool multi_instrument = (\w+)')
        match = pat.findall(setup)
        multi_instrument = True if match[0] == 'true' else False

        if multi_instrument:
            pat = re.compile(r'datafiles\s?\=\s?\{(.*?)\}',
                             flags=re.MULTILINE | re.DOTALL)
            match_inside_braces = pat.findall(setup)
            inside_braces = match_inside_braces[0]

            pat = re.compile(r'"(.*?)"')
            match = pat.findall(inside_braces)
            self.filename = match
            #     msg = f'Cannot find datafiles in {self.kima_setup}'
            #     raise ValueError(msg)

            pat = re.compile(r'load_multi\(datafiles,\s*"(.*?)"\s*,\s*(\d)')
            match = pat.findall(setup)
            if len(match) == 1:
                units, skip = match[0]
                self.units = units
                self.skip = int(skip)
            else:
                msg = f'Cannot find units and skip in {self.kima_setup}'
                raise ValueError(msg)

        else:
            if 'datafile = ""' in setup:
                self.filename = None
                return

            pat = re.compile(r'datafile\s?\=\s?"(.+?)"\s?;')
            match = pat.findall(setup)
            if len(match) == 1:
                self.filename = match
            else:
                msg = f'Cannot find datafile in {self.kima_setup}'
                raise ValueError(msg)

            pat = re.compile(r'load\(datafile,\s*"(.*?)"\s*,\s*(\d)')
            match = pat.findall(setup)
            if len(match) == 1:
                units, skip = match[0]
                self.units = units
                self.skip = int(skip)
            else:
                msg = f'Cannot find units and skip in {self.kima_setup}'
                raise ValueError(msg)

        # store that the model has been loaded
        self._loaded = True

    def load_OPTIONS(self):
        if not os.path.exists(self.OPTIONS_file):
            raise FileNotFoundError(f'Cannot find "{self.OPTIONS_file}"')

        options = open(self.OPTIONS_file).readlines()

        keys = list(self.OPTIONS.keys())
        i = 0
        for line in options:
            if line.strip().startswith('#'):
                continue
            val, _ = line.split('#')
            val = int(val)
            self.OPTIONS[keys[i]] = val
            i += 1

        try:
            with open('.KIMATHREADS') as f:
                self.threads = int(f.read().strip())
        except FileNotFoundError:
            pass

    def save(self):
        """ Save this model to the OPTIONS file and the kima_setup file """
        self._save_OPTIONS()

        kima_setup_f = os.path.join(self.directory, self.kima_setup)
        with open(kima_setup_f, 'w') as f:
            f.write('#include "kima.h"\n\n')
            self._write_settings(f)
            self._write_constructor(f)
            self._inside_constructor(f)
            self._start_main(f)
            self._set_data(f)
            self._write_sampler(f)
            self._end_main(f)
            f.write('\n\n')

    # the following are helper methods to fill parts of the kima_setup file
    def _write_settings(self, file):
        def r(val): return 'true' if val else 'false'
        cb = 'const bool'
        file.write(
            f'{cb} obs_after_HARPS_fibers = {r(self.obs_after_HARPS_fibers)};\n')
        file.write(f'{cb} GP = {r(self.GP)};\n')
        file.write(f'{cb} MA = {r(self.MA)};\n')
        file.write(f'{cb} hyperpriors = {r(self.hyperpriors)};\n')
        file.write(f'{cb} trend = {r(self.trend)};\n')
        file.write(f'{cb} multi_instrument = {r(self.multi_instrument)};\n')
        file.write(f'{cb} known_object = {r(self.known_object)};\n')
        file.write('\n')

    def _write_constructor(self, file):
        def r(val): return 'true' if val else 'false'
        file.write(
            f'RVmodel::RVmodel():fix({r(self.fix_Np)}),npmax({self.max_Np})\n')

    def _inside_constructor(self, file):
        file.write('{\n')

        def write_prior_2(name, sets, add_conditional=False):
            s = 'c->' if add_conditional else ''
            return s + f'{name} = make_prior<{sets[1]}>({sets[2]}, {sets[3]});\n'
        def write_prior_1(name, sets, add_conditional=False):
            s = 'c->' if add_conditional else ''
            return s + f'{name} = make_prior<{sets[1]}>({sets[2]});\n'

        got_conditional = False
        for name, sets in self.priors.items():
            # print(name, sets)
            if not sets[0]:  # if not default prior
                if name in self._planet_priors:
                    if not got_conditional:
                        file.write(
                            f'auto c = planets.get_conditional_prior();\n')
                        got_conditional = True
                    if sets[1] == 'Fixed':
                        file.write(write_prior_1(name, sets, True))
                    else:
                        file.write(write_prior_2(name, sets, True))
                else:
                    if sets[1] == 'Fixed':
                        file.write(write_prior_1(name, sets, False))
                    else:
                        file.write(write_prior_2(name, sets, False))

        file.write('\n}\n')
        file.write('\n')

    def _write_sampler(self, file):
        file.write('\tSampler<RVmodel> sampler = setup<RVmodel>(argc, argv);\n')
        file.write('\tsampler.run(50);\n')

    def _set_data(self, file):
        if self.filename is None:
            file.write(f'\tdatafile = "";\n')
            file.write(f'\tload(datafile, "{self.units}", {self.skip});\n')
            return

        if self.multi_instrument:
            files = [f'"{datafile}"' for datafile in self.filename]
            files = ', '.join(files)
            file.write(f'\tdatafiles = {{ {files} }};\n')
            file.write(
                f'\tload_multi(datafiles, "{self.units}", {self.skip});\n')
        else:
            file.write(f'\tdatafile = "{self.filename[0]}";\n')
            file.write(f'\tload(datafile, "{self.units}", {self.skip});\n')

    def _start_main(self, file):
        file.write('int main(int argc, char** argv)\n')
        file.write('{\n')

    def _end_main(self, file):
        file.write('\treturn 0;\n')
        file.write('}\n')
    ##

    def _save_OPTIONS(self):
        """ Save sampler settings to a OPTIONS file """
        opt = list(self.OPTIONS.values())
        options_f = os.path.join(self.directory, self.OPTIONS_file)
        with open(options_f, 'w') as file:
            file.write('# File containing parameters for DNest4\n')
            file.write(
                '# Put comments at the top, or at the end of the line.\n')
            file.write(f'{opt[0]}\t# Number of particles\n')
            file.write(f'{opt[1]}\t# new level interval\n')
            file.write(f'{opt[2]}\t# save interval\n')
            file.write(
                f'{opt[3]}\t# threadSteps: number of steps each thread does independently before communication\n')
            file.write(f'{opt[4]}\t# maximum number of levels\n')
            file.write(f'{opt[5]}\t# Backtracking scale length (lambda)\n')
            file.write(
                f'{opt[6]}\t# Strength of effect to force histogram to equal push (beta)\n')
            file.write(f'{opt[7]}\t# Maximum number of saves (0 = infinite)\n')
            file.write('    # (optional) samples file\n')
            file.write('    # (optional) sample_info file\n')
            file.write('    # (optional) levels file\n')

    def run(self):
        cmd = f'kima-run {self.directory}'
        out = subprocess.check_call(cmd.split())
