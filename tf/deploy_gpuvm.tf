###############################################################################
#
# Terraform: Deploy GPU-enabled Virtual Machine in Azure
#
# Reference: https://docs.microsoft.com/en-us/azure/developer/terraform/create-linux-virtual-machine-with-infrastructure
#
###############################################################################

###############################################################################
#
# Azure: Create network components
#
###############################################################################

# Virtual network
resource "azurerm_virtual_network" "vnet" {
    name                    = "vnet"
    address_space           = ["10.0.0.0/16"]
    location                = azurerm_resource_group.rg.location
    resource_group_name     =  azurerm_resource_group.rg.name
}

# Subnet
resource "azurerm_subnet" "subnet" {
    name                    = "subnet"
    resource_group_name     = azurerm_resource_group.rg.name
    virtual_network_name    = azurerm_virtual_network.vnet.name
    address_prefixes        = ["10.0.1.0/24"]
}

# Public IPs
resource "azurerm_public_ip" "publicip" {
    name                    = "publicip"
    location                = azurerm_resource_group.rg.location
    resource_group_name     = azurerm_resource_group.rg.name
    allocation_method       = "Dynamic"
}

# Network Security Group and rule
resource "azurerm_network_security_group" "nsg" {
    name                = "nsg"
    location            = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    
    security_rule {
        name                       = "SSH"
        priority                   = 1001
        direction                  = "Inbound"
        access                     = "Allow"
        protocol                   = "Tcp"
        source_port_range          = "*"
        destination_port_range     = "22"
        source_address_prefix      = "*"
        destination_address_prefix = "*"
    }
}

# Network interface configuration
resource "azurerm_network_interface" "nic" {
    name                = "nic"
    location            = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    
    ip_configuration {
        name                          = "nic-configuration"
        subnet_id                     = azurerm_subnet.subnet.id
        private_ip_address_allocation = "Dynamic"
        public_ip_address_id          = azurerm_public_ip.publicip.id
    }
}

###############################################################################
#
# Azure: Security
#
###############################################################################
# Connect the security group to the network interface
resource "azurerm_network_interface_security_group_association" "nicsga" {
    network_interface_id      = azurerm_network_interface.nic.id
    network_security_group_id = azurerm_network_security_group.nsg.id
}

# Generate random text for a unique storage account name
resource "random_id" "id" {
    keepers = {
        # Generate a new ID only when a new resource group is defined
        resource_group = azurerm_resource_group.rg.name
    }
    byte_length = 8
}

# Create storage account for boot diagnostics
resource "azurerm_storage_account" "gpuvm_sa" {
    name                     = "diag${random_id.id.hex}"
    location                 = azurerm_resource_group.rg.location
    resource_group_name      = azurerm_resource_group.rg.name
    account_tier             = "Standard"
    account_replication_type = "LRS"
}

# Create (and display) an SSH key
resource "tls_private_key" "ssh_key" {
    algorithm = "RSA"
    rsa_bits  = 4096
}

###############################################################################
#
# Azure: GPU-enabled Virtual Machine
#
# Notes:
#   - UbuntuServer 20.04 doesn't seem to be available
#
###############################################################################

# Virtual machine
resource "azurerm_linux_virtual_machine" "gpuvm" {
    name                    = "cuda_testing_gpuvm"
    location                = azurerm_resource_group.rg.location
    resource_group_name     = azurerm_resource_group.rg.name
    network_interface_ids   = [azurerm_network_interface.nic.id]
    size                    = "Standard_NC6s_v3"

    os_disk {
        name                    = "OsDisk"
        caching                 = "ReadWrite"
        storage_account_type    = "Standard_LRS"
    }

    source_image_reference {
        publisher               = "Canonical"
        offer                   = "UbuntuServer"
        sku                     = "18.04-LTS"
        version                 = "latest"
    }

    computer_name                   = "pyramidtestvm"
    admin_username                  = "pyramidtestuser"
    disable_password_authentication = true
    
    admin_ssh_key {
        username   = "pyramidtestuser"
        public_key = tls_private_key.ssh_key.public_key_openssh
    }
    
    boot_diagnostics {
        storage_account_uri = azurerm_storage_account.gpuvm_sa.primary_blob_endpoint
    }
}

# Daily shutdown - for safety in case the VM is left on
# See https://jackstromberg.com/2017/01/list-of-time-zones-consumed-by-azure/
resource "azurerm_dev_test_global_vm_shutdown_schedule" "shutdown" {
    virtual_machine_id      = azurerm_linux_virtual_machine.gpuvm.id
    location                = azurerm_resource_group.rg.location
    enabled                 = true
    
    daily_recurrence_time   = "1930"
    timezone                = "GMT Standard Time"
    
    notification_settings {
        enabled             = false
        time_in_minutes     = 30
    }
}

# Virtual machine extension to install the NVidia GPU drivers
resource "azurerm_virtual_machine_extension" "NvidiaGpuDriverLinux" {
    name                        = "NvidiaGpuDriverLinux"
    virtual_machine_id          = azurerm_linux_virtual_machine.gpuvm.id
    publisher                   = "Microsoft.HpcCompute"
    type                        = "NvidiaGpuDriverLinux"
    type_handler_version        = "1.2"
    auto_upgrade_minor_version  = true
}

# Attached disk - needed for docker build as OS disk is only 32GB
#resource "azurerm_managed_disk" "build" {
#    name                 = "pyramidtestvm-build"
#    location             = azurerm_resource_group.rg.location
#    resource_group_name  = azurerm_resource_group.rg.name
#    storage_account_type = "Standard_LRS"
#    create_option        = "Empty"
#    disk_size_gb         = 1024
#}
#
#resource "azurerm_virtual_machine_data_disk_attachment" "build" {
#    managed_disk_id    = azurerm_managed_disk.build.id
#    virtual_machine_id = azurerm_linux_virtual_machine.gpuvm.id
#    lun                = "10"
#    caching            = "ReadWrite"
#}