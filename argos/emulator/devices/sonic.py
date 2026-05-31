import logging
from numpy import random
import pandas
import os

print("cwd:", os.getcwd())
from argos.utils.logging import get_logger as argos_get_logger

class demoDevice:


    def __init__(self,deviceName,frequency,duration):
        """
            Initializes a demo device with the fields.

        Parameters
        ----------
        deviceConfiguration : dict
                A configuration of the device:


        """
        self.logger = argos_get_logger(self)
        self.deviceName = deviceName
        self.frequency = frequency
        self.duration = duration

    @property
    def delay_s(self):
        return pandas.Timedelta('%.3fs' % (1/float(self.frequency))).value/1e9

    def stream_messages(self):
        import time
        import pandas
        from numpy import random

        u = v = w = T = 0

        startDate = pandas.Timestamp.now()
        endTime = startDate + pandas.to_timedelta(self.duration)

        timeDelta = pandas.Timedelta('%.3fs' % (1 / float(self.frequency)))

        self.logger.execution(
            f"Streaming from {startDate} to {endTime} at interval {timeDelta}"
        )

        ts = startDate

        while ts <= endTime:
            u = self.u_min + (self.u_max - self.u_min) * random.rand()
            v = self.v_min + (self.v_max - self.v_min) * random.rand()
            w = self.w_min + (self.w_max - self.w_min) * random.rand()
            T = self.T_min + (self.T_max - self.T_min) * random.rand()

            yield {
                "datetime": ts,
                "u": u,
                "v": v,
                "w": w,
                "t": T
            }

            ts += timeDelta
            time.sleep(timeDelta.total_seconds())
    def emmitMessage(self):
        pass

class deviceSonic(demoDevice):

    u_max = 6
    u_min = -6
    v_max = 6
    v_min = -6
    w_max = 0.3
    w_min = -0.3
    T_max = 40
    T_min = 20

    def __init__(self,deviceName,frequency,duration):
        super().__init__(deviceName,frequency,duration)


    def createMessages(self):
        u = 0
        v = 0
        w = 0
        T = 0
        self.logger.execution("------------ Start -----------------")
        self.logger.debug(f"Frequency {self.frequency} duration {self.duration}")
        startDate = pandas.Timestamp.now()
        endTime   = startDate + pandas.to_timedelta(self.duration)
        timeDelta = pandas.Timedelta('%.3fs' % (1/float(self.frequency)))

        self.logger.execution(f"From {startDate} to {endTime} at frequency {timeDelta}")

        dateRange = pandas.date_range(startDate,endTime,freq=timeDelta)
        self.logger.execution(f"Creating {len(dateRange)} messages")
        for ts in dateRange:
            u = self.u_min + (self.u_max - self.u_min) * random.rand()
            v = self.v_min + (self.v_max - self.v_min) * random.rand()
            w = self.w_min + (self.w_max - self.w_min) * random.rand()
            T = self.T_min + (self.T_max - self.T_min) * random.rand()
            self.messages.append(dict(datetime=ts.value,u=u,v=v,w=w,t=T))

        self.logger.execution("------------ End -----------------")
