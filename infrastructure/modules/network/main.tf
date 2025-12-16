resource "yandex_vpc_network" "mlops_net" {
  name        = "mlops-credit-scoring-net"
  description = "VPC network for credit scoring MLOps PJ"
}

resource "yandex_vpc_subnet" "subnet_a" {
  name           = "subnet-ru-central1-a"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.mlops_net.id
  v4_cidr_blocks = ["10.1.0.0/24"]
}

resource "yandex_vpc_subnet" "subnet_b" {
  name           = "subnet-ru-central1-b"
  zone           = "ru-central1-b"
  network_id     = yandex_vpc_network.mlops_net.id
  v4_cidr_blocks = ["10.2.0.0/24"]
}

resource "yandex_vpc_subnet" "subnet_d" {
  name           = "subnet-ru-central1-d"
  zone           = "ru-central1-d"
  network_id     = yandex_vpc_network.mlops_net.id
  v4_cidr_blocks = ["10.3.0.0/24"]
}