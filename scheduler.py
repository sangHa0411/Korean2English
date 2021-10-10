
class Scheduler :
    def __init__(self, d_model, init_lr, warmup_steps) :
        self.d_model = d_model
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

    def __call__(self, epoch) :
        step_num = epoch + 1
        val1 = self.d_model ** (-0.5)
        arg1 = step_num ** (-0.5)
        arg2 = (self.warmup_steps ** (-1.5)) * step_num
        val2 = min(arg1 , arg2) 
        return (val1 * val2) / self.init_lr
