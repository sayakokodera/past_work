# -*- coding: utf-8 -*-
import datetime
"""
#================ Date Formatting for File Names ================#
Example usage:
    (1) dateformatter = DateFormatter()
    (2) f_date = dateformatter.get_date_str()
"""

class DateTimeFormatter():
    
    def __init__(self):
        r""" Constructor
        """
        super().__init__()
    
    # check the digits of the month/day
    def check_digit(self, input_int):
        r""" Check the number of digits of the date/time information and add 0, when it is with 1 digit
        (e.g. now.month = 7 -> month_str = 07)
        Parameter
        ---------
            input_int : int
                The input int (e.g. self.month) for checking the number of digit
            
        Return
        ------
            output_str : str
                The output string in the proper form (e.g. 7 -> 07)
        """
        # when day or month is only with 1 digit -> add 0
        if len(str(input_int)) < 2:
            output_str = '0{}'.format(input_int)
        else:
            output_str = str(input_int)
        return output_str
    

    def get_date_str(self):
        r""" Convert the current data information (i.e. self.now) into the proper
        string for file names
        Return
        ------
            today : str
                containing info on year-> month -> day
        """
        # Get now
        now = datetime.datetime.now()
        # remove the first "20" from year: e.g. 2020 -> 20
        year_str = str(now.year)[2:]
        # check digits
        month_str = self.check_digit(now.month)
        day_str = self.check_digit(now.day)
        # Format the output string
        today = '{}{}{}'.format(year_str, month_str, day_str) 
        
        return today
    
    def get_time_str(self):
        r""" Convert the current time information (i.e. self.now) into the proper
        string for file names
        Return
        ------
            curr_time : str
                containing info on year-> month -> day -> hour -> minute -> second
        """
        # Get now
        now = datetime.datetime.now()
        # check digits
        month_str = self.check_digit(now.month)
        day_str = self.check_digit(now.day)
        hour_str = self.check_digit(now.hour)
        min_str = self.check_digit(now.minute)
        sec_str = self.check_digit(now.second)
        # Format the output string
        curr_time = '{}{}{}_{}h{}m{}s'.format(now.year, month_str, day_str, hour_str, 
                                              min_str, sec_str)
        
        return curr_time
    
    
    