from dataclasses import dataclass


@dataclass
class VastOffer:
    """
    A vast.ai offer data class.
    An offer is a machine that can be rented by the user.
    """
    index: int
    id: str
    cuda: str
    num: str
    model: str
    pcie: str
    cpu_ghz: str
    vcpus: str
    ram: str
    disk: str
    price: str
    dlp: str
    dlp_per_dollar: str
    score: str
    nv_driver: str
    net_up: str
    net_down: str
    r: str
    max_days: str
    mach_id: str
    status: str
    ports: str
    country: str

    def __str__(self):
        return self.tostring(self.index)

    def tostring(self, i):
        return f'[{i + 1:02}] - {self.model} - {self.num} - {self.dlp} - {self.net_down} Mbps - {self.price} $/hr - {self.dlp_per_dollar} DLP/HR'
