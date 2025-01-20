# CHANGELOG


## v0.1.4 (2025-01-20)

### Bug Fixes

* fix: adding highpass filtering + windowing, adding Eryn

Merge pull request #1 from pywavelet/ollie_dev ([`08ad07e`](https://github.com/pywavelet/case_studies.gaps/commit/08ad07eec587f19fffa93bbfc2f37711505e181b))

### Unknown

* considering small gap, rather than two ([`cfcd853`](https://github.com/pywavelet/case_studies.gaps/commit/cfcd8536f1783bb00c938a02ce78b48071b88a3d))

* adding in burnin feature ([`cebc1c4`](https://github.com/pywavelet/case_studies.gaps/commit/cebc1c46c1a0cfb4248fe3021b2287dfcd9bdbe4))

* allowing for filtering + window ([`35dc907`](https://github.com/pywavelet/case_studies.gaps/commit/35dc907d716d9fc462f6bc59f0bf8cfa16ccf4e2))

* eryn priors with extra burnin phase

Added in eryn priors and included an extra step to allow for burnin.

The format for eryn (final .hdf5) is a little different from the .hdf5 file
that is produced using emcee. This makes it incompatible with the package _arviz_.

Need to resolve this, or change plotting utilities if we wish to change to eryn. ([`52e5625`](https://github.com/pywavelet/case_studies.gaps/commit/52e56253033c4bd72bcf3f3e3649e79ee4f1a578))

* testing no gaps no noise ([`4528af2`](https://github.com/pywavelet/case_studies.gaps/commit/4528af2eff9fa1d79229bfa454f53c0a81312077))

* pip install uses eryn ([`b37f991`](https://github.com/pywavelet/case_studies.gaps/commit/b37f9911759a18634bc86b38fe0701a42c9d4f5e))


## v0.1.3 (2025-01-20)

### Chores

* chore(release): 0.1.3 ([`910e6cf`](https://github.com/pywavelet/case_studies.gaps/commit/910e6cf0cf8732d5db8abf8fa6fcd37eeeb7bfe0))

### Unknown

* Merge branch 'main' of github.com:pywavelet/case_studies.gaps ([`42706a4`](https://github.com/pywavelet/case_studies.gaps/commit/42706a4fef1505995c14ee697d7947be72722a7a))


## v0.1.2 (2025-01-20)

### Bug Fixes

* fix: add gap corners ([`68a540f`](https://github.com/pywavelet/case_studies.gaps/commit/68a540fa76336d1180430ddfa3347fc837961738))

### Chores

* chore(release): 0.1.2 ([`3cf1849`](https://github.com/pywavelet/case_studies.gaps/commit/3cf18491c7a6a7449a7496f9e03b1d8d1187195d))

### Unknown

* Merge remote-tracking branch 'origin/main' ([`0ae04f4`](https://github.com/pywavelet/case_studies.gaps/commit/0ae04f48ac48112f2baecc2c8beb6d24e3899d8b))


## v0.1.1 (2025-01-20)

### Bug Fixes

* fix: add lnl tests ([`7a4b703`](https://github.com/pywavelet/case_studies.gaps/commit/7a4b703166a976291c2108b333a1acf4c50627aa))

* fix: remove harcoded highpass filter, change hdatawavelet construction (signal(t,f) + noise(t,f)) rather than converting data(t) = noise(t) + signal(t)) ([`e4f1a8c`](https://github.com/pywavelet/case_studies.gaps/commit/e4f1a8c560a6bc66d58eca0e793cb25a7f855c2a))

### Chores

* chore(release): 0.1.1 ([`14225b4`](https://github.com/pywavelet/case_studies.gaps/commit/14225b4bdf6cea95bffb9dd4cabaf1a599014efc))

### Unknown

* change __ to _ for private vars ([`9da28fe`](https://github.com/pywavelet/case_studies.gaps/commit/9da28fed2419b9a20be30026bcd67ffffeaf235b))

* add log when noise being generated ([`a89c215`](https://github.com/pywavelet/case_studies.gaps/commit/a89c21556b04d30efe909b16aafa7916ba387ae6))

* add LnL amplitude check ([`a166fe3`](https://github.com/pywavelet/case_studies.gaps/commit/a166fe3db3fa2adf7cbd99655f1901c9e4bc57ee))


## v0.1.0 (2025-01-16)

### Chores

* chore(release): 0.1.0 ([`32e4e3c`](https://github.com/pywavelet/case_studies.gaps/commit/32e4e3c435093f6ce908bec3cab901a994d7f531))

### Unknown

* Merge branch 'main' of github.com:pywavelet/case_studies.gaps into main ([`c53e3eb`](https://github.com/pywavelet/case_studies.gaps/commit/c53e3eb5634ed46a0bb459da19a3814cb9d3bdb8))


## v0.0.4 (2025-01-15)

### Chores

* chore(release): 0.0.4 ([`8801219`](https://github.com/pywavelet/case_studies.gaps/commit/8801219cc64e0cb3c73419e782431283fcbce64c))

### Features

* feat: add bilby sampler ([`262b4eb`](https://github.com/pywavelet/case_studies.gaps/commit/262b4ebc4824ddcccf69271f779fe582c5f017b4))

### Unknown

* Merge branch 'main' of github.com:pywavelet/case_studies.gaps into main ([`fb5526d`](https://github.com/pywavelet/case_studies.gaps/commit/fb5526dcaa2fde817dd75c6ec537a2f957b6c868))

* Update docs.yml ([`0a28a6a`](https://github.com/pywavelet/case_studies.gaps/commit/0a28a6a1057641c548c1a27dde1ce8a6e0c88f0b))


## v0.0.3 (2024-12-12)

### Bug Fixes

* fix: add frange ([`9acb21e`](https://github.com/pywavelet/case_studies.gaps/commit/9acb21eccfe28a886c48372d1fe9a0054a70c6b4))

* fix: add badges to readme ([`7b8b2b3`](https://github.com/pywavelet/case_studies.gaps/commit/7b8b2b34b56bc7ffe7ec2a7c94b5eec061b2cf17))

### Chores

* chore(release): 0.0.3 ([`7cecccc`](https://github.com/pywavelet/case_studies.gaps/commit/7cecccc115d74626359531091139d971ba3db1b7))

### Unknown

* Merge branch 'main' of github.com:pywavelet/case_studies.gaps into main ([`6d06075`](https://github.com/pywavelet/case_studies.gaps/commit/6d0607565f7817c805ca252c908649c91db8ad48))


## v0.0.2 (2024-12-12)

### Chores

* chore(release): 0.0.2 ([`71676bb`](https://github.com/pywavelet/case_studies.gaps/commit/71676bb40be374f15a0c215bf220dff9b4db9b0f))

### Unknown

* fix pytest CI ([`b2e622d`](https://github.com/pywavelet/case_studies.gaps/commit/b2e622d90595e95f1d41681c297ac80b55bf0551))

* Merge branch 'main' of github.com:pywavelet/case_studies.gaps into main ([`00a9450`](https://github.com/pywavelet/case_studies.gaps/commit/00a945053c28cb5335a33659ce7fad7af4487db1))


## v0.0.1 (2024-12-12)

### Bug Fixes

* fix: remove git dependance for pywavelet package ([`e81ec2f`](https://github.com/pywavelet/case_studies.gaps/commit/e81ec2f6efc7ac5d4fbd23f4787f44bc93d81e5e))

* fix: init repo ([`ce18d51`](https://github.com/pywavelet/case_studies.gaps/commit/ce18d51c21a5317e08f4671f24807bae5de4fa34))

### Chores

* chore(release): 0.0.1 ([`d1e964d`](https://github.com/pywavelet/case_studies.gaps/commit/d1e964dd20c11332a5413cf55d2280f5e10ecfb1))

### Unknown

* Initial commit ([`cee6ff7`](https://github.com/pywavelet/case_studies.gaps/commit/cee6ff7f33ba99057e4439071e3267bd6b9bbcf9))
