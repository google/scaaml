"""Context manager for the scope."""
import chipwhisperer as cw


class SScope:
    """Scope context manager."""
    def __init__(self, gain: int, samples: int, offset: int, clock: int,
                 sample_rate: str, **_):
        """Create scope context.

        Args:
          gain: Gain of the scope.
          samples: How many points to sample (length of the capture).
          offset: Number of samples to wait before starting recording data.
          clock: CLKGEN output frequency (in Hz).
          sample_rate: Clock source for cw.ClockSettings.adc_src.
          _: SScope is expected to be initialized using the capture_info
            dictionary which may contain extra keys (additional information
            about the capture; the capture_info dictionary is saved in the
            info file of the dataset). Thus we can ignore the rest of keyword
            arguments.

        Expected use:
          capture_info = {
              'gain': gain,
              'samples': samples,
              'offset': offset,
              'clock': clock,
              'sample_rate': sample_rate,
              'other_information': 'Can also be present.',
          }
          with SScope(**capture_info) as scope:
              # Use the scope object.
        """
        self._scope = None
        self._gain = gain
        self._samples = samples
        self._offset = offset
        self._clock = clock
        self._sample_rate = sample_rate
        self._basic_mode = "rising_edge"
        self._triggers = "tio4"
        self._freq_ctr_src = "clkgen"
        self._presamples = 0

    def __enter__(self):
        """Create scope context.

        Args:
          gain: Gain of the scope.
          samples: How many points to sample (length of the capture).
          offset: Number of samples to wait before starting recording data.
          clock: CLKGEN output frequency (in Hz).
          sample_rate: Clock source for cw.ClockSettings.adc_src.
        """
        assert self._scope is None  # Do not allow nested with.
        cwscope = cw.scope()
        cwscope.gain.db = self._gain
        max_samples = cwscope.adc.oa.hwInfo.maxSamples()
        if self._samples > max_samples and cwscope.adc.oa.hwInfo.is_cw1200():
            cwscope.adc.stream_mode = True
        cwscope.adc.samples = self._samples
        cwscope.adc.offset = self._offset
        cwscope.adc.basic_mode = self._basic_mode
        cwscope.clock.clkgen_freq = self._clock
        cwscope.trigger.triggers = self._triggers
        cwscope.clock.adc_src = self._sample_rate
        cwscope.clock.freq_ctr_src = self._freq_ctr_src
        cwscope.adc.presamples = self._presamples
        self._scope = cwscope
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Safely close all resources.

        Args:
          exc_type: None if no exception, otherwise the exception type.
          exc_value: None if no exception, otherwise the exception value.
          exc_tb: None if no exception, otherwise the traceback.
        """
        self._scope.dis()
        self._scope = None

    @property
    def scope(self):
        """Returns the scope object."""
        return self._scope
