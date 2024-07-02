from dataclasses import dataclass


@dataclass
class VastInstance:
    """
    A vast.ai instance data class.
    An instance is a machine rente by the user. (running, stopped, launching, etc.)
    It contains private information for launching the data.
    """
    index: int
    id: str
    machine: str
    status: str
    num: str
    model: str
    util: str
    vcpus: str
    ram: str
    storage: str
    sshaddr: str
    sshport: str
    price: str
    image: str
    netup: str
    netdown: str
    r: str
    label: str
    age: str

    def __str__(self):
        return f'{self.index + 1} - {self.model} - {self.num} - {self.netdown} Mbps - {self.price} $/hr - {self.status}'

    @property
    def ip(self):
        return self.sshaddr

    @property
    def port(self):
        return self.sshport