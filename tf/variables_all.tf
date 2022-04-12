###############################################################################
#
# Azure: Variables
#
###############################################################################
# PYRAMID Subscription ID
variable "subscription_id" {
  default = "0a755043-43e9-47f4-83c4-ea715af7e9a6"
}

# Newcastle University Tenant ID
variable "tenant_id" {
  default = "9c5012c9-b616-44c2-a917-66814fbe3e87"
}

variable "resource_group_name" {
  default = "cuda-testing-rg"
}

# Requires a UK South location for GPU-enabled VMs
variable "resource_group_location" {
  default = "uksouth"
}


###############################################################################
#
# Azure: Information about the project
#
###############################################################################
variable "project-name" {
  type = string
  default = "PYRAMID CUDA Testing VMs"
}

variable "project-investigators" {
  type = string
  default = "Hayley Fowler, Elizabeth Lewis"
}

variable "project-contributors" {
  type = string
  default = "Robin Wardle"
}